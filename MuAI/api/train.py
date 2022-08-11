import typing
import tensorflow as tf
from ..api.utils import distribution_strategy


MAE = tf.keras.losses.MeanAbsoluteError()
BCE = tf.keras.losses.BinaryCrossentropy()


class Trainer:
    def __init__(
        self, data, save_dir, epochs=2, batch_size=32, verbose=1, device="gpu"
    ):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.save_dir = save_dir
        self.strategy = distribution_strategy(device)

    def build(self, *models, optimizer: typing.Callable = lambda: "adadelta", **kwargs):
        self.model = {}
        self.g_optimizer = {}
        self.d_optimizer = {}

        with self.strategy.scope():
            for model in models:
                model.build(**kwargs)
                self.model[model.name] = model
                self.g_optimizer[model.name] = optimizer()
                self.d_optimizer[model.name] = optimizer()

    def train(self):
        @tf.function
        def train_step():
            ya, yi, yf = self.data.get_batch(self.batch_size)
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                xa = self.model["Acapella"].generator(yi)
                xi = self.model["Instrumental"].generator(ya)
                xf = self.model["Intermixer"].generator(xa, xi)

                da_real = self.model["Acapella"].discriminator(ya)
                di_real = self.model["Instrumental"].discriminator(yi)
                df_real = self.model["Intermixer"].discriminator(xf)

                da_fake = self.model["Acapella"].discriminator(xa)
                di_fake = self.model["Instrumental"].discriminator(xi)
                df_fake = self.model["Intermixer"].discriminator(xf)

                Ga_loss = 0.5 * (BCE(xa, ya) + MAE(da_fake, tf.ones_like(da_fake)))
                Gi_loss = 0.5 * (BCE(xi, yi) + MAE(di_fake, tf.ones_like(di_fake)))
                Gf_loss = 0.5 * (BCE(xf, yf) + MAE(df_fake, tf.ones_like(df_fake)))
                G_loss = Ga_loss + Gi_loss + Gf_loss

                Da_loss = 0.5 * (
                    MAE(da_real, tf.ones_like(da_real))
                    + MAE(da_fake, tf.zeros_like(da_fake))
                )
                Di_loss = 0.5 * (
                    MAE(di_real, tf.ones_like(di_real))
                    + MAE(di_fake, tf.zeros_like(di_fake))
                )
                Df_loss = 0.5 * (
                    MAE(df_real, tf.ones_like(df_real))
                    + MAE(df_fake, tf.zeros_like(df_fake))
                )
                D_loss = Da_loss + Di_loss + Df_loss

            g_grads = g_tape.gradient(
                G_loss,
                self.model["Acapella"].trainable_variables
                + self.model["Instrumental"].trainable_variables
                + self.model["Intermixer"].trainable_variables,
            )
            d_grads = d_tape.gradient(
                D_loss,
                self.model["Acapella"].trainable_variables
                + self.model["Instrumental"].trainable_variables
                + self.model["Intermixer"].trainable_variables,
            )

            self.g_optimizer["Acapella"].apply_gradients(
                zip(g_grads, self.model["Acapella"].trainable_variables)
            )
            self.g_optimizer["Instrumental"].apply_gradients(
                zip(g_grads, self.model["Instrumental"].trainable_variables)
            )
            self.g_optimizer["Intermixer"].apply_gradients(
                zip(g_grads, self.model["Intermixer"].trainable_variables)
            )

            self.d_optimizer["Acapella"].apply_gradients(
                zip(d_grads, self.model["Acapella"].trainable_variables)
            )
            self.d_optimizer["Instrumental"].apply_gradients(
                zip(d_grads, self.model["Instrumental"].trainable_variables)
            )
            self.d_optimizer["Intermixer"].apply_gradients(
                zip(d_grads, self.model["Intermixer"].trainable_variables)
            )
