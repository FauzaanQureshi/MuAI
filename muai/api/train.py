import typing
import tensorflow as tf

from .config import config as CONFIG
from .utils import distribution_strategy


MAE = tf.keras.losses.MeanAbsoluteError()
BCE = tf.keras.losses.BinaryCrossentropy()


class Trainer:
    def __init__(
        self,
        data,
        save_dir,
        epochs=2,
        batch_size=32,
        verbose=CONFIG.verbose,
        device=CONFIG.device,
    ):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.save_dir = save_dir
        self.strategy = distribution_strategy(device)

    def build(self, *models, **kwargs):
        self.model = {}
        self.g_optimizer = CONFIG.generator.optimizer(lr=CONFIG.generator.lr)
        self.d_optimizer = CONFIG.discriminator.optimizer(lr=CONFIG.discriminator.lr)

        with self.strategy.scope():
            for model in models:
                model.build(**kwargs)
                self.model[model.name] = model

    def train(self):
        @tf.function
        def train_step():
            # ----------------------------------------------------------------
            # Train block 1                                                  |
            # ----------------------------------------------------------------
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

            self.model["Acapella"].freeze("discriminator")
            self.model["Instrumental"].freeze("discriminator")
            self.model["Intermixer"].freeze("discriminator")
            g_grads = g_tape.gradient(
                G_loss,
                self.model["Acapella"].trainable_variables
                + self.model["Instrumental"].trainable_variables
                + self.model["Intermixer"].trainable_variables,
            )
            self.g_optimizer.apply_gradients(
                zip(
                    g_grads,
                    (
                        self.model["Acapella"].trainable_variables
                        + self.model["Instrumental"].trainable_variables
                        + self.model["Intermixer"].trainable_variables
                    ),
                )
            )
            self.model["Acapella"].unfreeze("discriminator")
            self.model["Instrumental"].unfreeze("discriminator")
            self.model["Intermixer"].unfreeze("discriminator")

            self.model["Acapella"].freeze("generator")
            self.model["Instrumental"].freeze("generator")
            self.model["Intermixer"].freeze("generator")
            d_grads = d_tape.gradient(
                D_loss,
                self.model["Acapella"].trainable_variables
                + self.model["Instrumental"].trainable_variables
                + self.model["Intermixer"].trainable_variables,
            )
            self.d_optimizer.apply_gradients(
                zip(
                    d_grads,
                    (
                        self.model["Acapella"].trainable_variables
                        + self.model["Instrumental"].trainable_variables
                        + self.model["Intermixer"].trainable_variables
                    ),
                )
            )
            self.model["Acapella"].unfreeze("generator")
            self.model["Instrumental"].unfreeze("generator")
            self.model["Intermixer"].unfreeze("generator")

            # ----------------------------------------------------------------
            # Train block 2                                                  |
            # ----------------------------------------------------------------
            ya, yi, _ = self.data.get_batch(self.batch_size)
            with tf.GradientTape() as ga_tape, tf.GradientTape() as gi_tape:
                xa = self.model["Acapella"].generator(yi)
                xi_2 = self.model["Instrumental"].generator(xa)

                xi = self.model["Instrumental"].generator(ya)
                xa_2 = self.model["Acapella"].generator(xi)

                Ga_loss = MAE(xa_2, ya)
                Gi_loss = MAE(xi_2, yi)
                G_loss = Ga_loss + Gi_loss

            ga_grads = ga_tape.gradient(
                G_loss, self.model["Acapella"].generator.trainable_variables
            )
            self.g_optimizer.apply_gradients(
                zip(ga_grads, (self.model["Acapella"].generator.trainable_variables))
            )
            gi_grads = gi_tape.gradient(
                G_loss, self.model["Instrumental"].generator.trainable_variables
            )
            self.g_optimizer.apply_gradients(
                zip(
                    gi_grads, (self.model["Instrumental"].generator.trainable_variables)
                )
            )
