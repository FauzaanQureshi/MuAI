from datetime import datetime
from tqdm.autonotebook import tqdm
import tensorflow as tf

from .config import config as CONFIG
from .utils import distribution_strategy


MAE = tf.keras.losses.MeanAbsoluteError(reduction="none")
BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")


class Trainer:
    def __init__(
        self,
        dataset,
        save_dir: str,
        epochs: int = None,
        steps_per_epoch: int = None,
        batch_size: int = None,
        verbose: int = None,
        device: str = None,
    ):
        """
        dataset: tf.data.Dataset object
        save_dir: Save model dir. Use CONFIG.save_dir only
        epochs: Number of times the dataset is iterated over
            default= CONFIG.epochs
        steps_per_epoch: No. of training steps in an epoch
            default= CONFIG.steps_per_epoch
        batch_size: dataset batch_size
            default= CONFIG.batch_size
        verbose: 0 = silent, 1 = progress bar, 2 = Info
            default= CONFIG.verbose
        device: Device to run Trainer on. Use CONFIG.device only
            default= CONFIG.device
        """
        if not epochs:
            epochs = CONFIG.epochs
        if not steps_per_epoch:
            steps_per_epoch = CONFIG.steps_per_epoch
        if not batch_size:
            batch_size = CONFIG.batch_size
        if not verbose:
            verbose = CONFIG.verbose
        if not device:
            device = CONFIG.device

        self.dataset = dataset
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.save_dir = save_dir
        self.strategy = distribution_strategy(device)
        self.device = device

    def build(self, acapella, instrumental, mixer, **kwargs):
        self.model = {
            "Acapella": acapella,
            "Instrumental": instrumental,
            "Intermixer": mixer,
        }
        self.metrics = {}
        self.optimizer = {}

        with self.strategy.scope():
            g_opt = lambda: CONFIG.generator.optimizer(**CONFIG.generator.opt_kwargs)
            d_opt = lambda: CONFIG.discriminator.optimizer(
                **CONFIG.discriminator.opt_kwargs
            )
            if not (acapella and instrumental and mixer):
                acapella.build(in_scope=True)
                instrumental.build(in_scope=True)
                mixer.build(in_scope=True)
            self.optimizer = dict(
                ag_opt=g_opt(),
                ig_opt=g_opt(),
                mg_opt=g_opt(),
                ad_opt=d_opt(),
                id_opt=d_opt(),
                md_opt=d_opt(),
            )
            self.metrics = dict(
                ag_loss=tf.keras.metrics.Mean("ag_loss", dtype=tf.float32),
                ig_loss=tf.keras.metrics.Mean("ig_loss", dtype=tf.float32),
                mg_loss=tf.keras.metrics.Mean("mg_loss", dtype=tf.float32),
                ad_loss=tf.keras.metrics.Mean("ad_loss", dtype=tf.float32),
                id_loss=tf.keras.metrics.Mean("id_loss", dtype=tf.float32),
                md_loss=tf.keras.metrics.Mean("md_loss", dtype=tf.float32),
                ag_acc=tf.keras.metrics.BinaryAccuracy("ag_acc", dtype=tf.float32),
                ad_acc=tf.keras.metrics.BinaryAccuracy("ad_acc", dtype=tf.float32),
                ig_acc=tf.keras.metrics.BinaryAccuracy("ig_acc", dtype=tf.float32),
                id_acc=tf.keras.metrics.BinaryAccuracy("id_acc", dtype=tf.float32),
                mg_acc=tf.keras.metrics.BinaryAccuracy("mg_acc", dtype=tf.float32),
                md_acc=tf.keras.metrics.BinaryAccuracy("md_acc", dtype=tf.float32),
            )
            self.checkpoint = tf.train.Checkpoint(
                ag_opt=self.optimizer["ag_opt"],
                ig_opt=self.optimizer["ig_opt"],
                mg_opt=self.optimizer["mg_opt"],
                ad_opt=self.optimizer["ad_opt"],
                id_opt=self.optimizer["id_opt"],
                md_opt=self.optimizer["md_opt"],
            )

    def train(self):
        acapella = self.model["Acapella"]
        instrumental = self.model["Instrumental"]
        mixer = self.model["Intermixer"]
        summary_writer = tf.summary.create_file_writer(
            CONFIG.log_dir + f"/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        @tf.function
        def train_step(data_iterator):
            def _step(*inputs):
                def block(ya, yi, ym):
                    _xa = acapella(yi)
                    _xi = instrumental(yi)
                    _xm = mixer((ya + yi) / 2)
                    # print("\n"*5+f"_xa.shape:{_xa.shape}\n_xm.shape:{_xm.shape}"+"\n"*5)
                    _xxm = mixer((_xa + _xi) / 2)

                    xa = acapella(_xi)
                    xi = instrumental(_xa)
                    xm = mixer((xa + xi) / 2)

                    dra = acapella.discriminator(ya)
                    dfa = acapella.discriminator(_xa)
                    dffa = acapella.discriminator(xa)

                    dri = instrumental.discriminator(yi)
                    dfi = instrumental.discriminator(_xi)
                    dffi = instrumental.discriminator(xi)

                    drm = mixer.discriminator(ym)
                    dfm = mixer.discriminator(_xm)
                    dffm1 = mixer.discriminator(_xxm)
                    dffm2 = mixer.discriminator(xm)

                    # Generator Adverserial Loss
                    l_dfa_g = BCE(tf.ones_like(dfa), dfa)
                    l_dffa_g = BCE(tf.ones_like(dffa), dffa)

                    l_dfi_g = BCE(tf.ones_like(dfi), dfi)
                    l_dffi_g = BCE(tf.ones_like(dffi), dffi)

                    l_dfm_g = BCE(tf.ones_like(dfm), dfm)
                    l_dffm1_g = BCE(tf.ones_like(dffm1), dffm1)
                    l_dffm2_g = BCE(tf.ones_like(dffm2), dffm2)

                    # Discriminator Adverserial Loss
                    l_dra_d = BCE(tf.ones_like(dra), dra)
                    l_dfa_d = BCE(tf.zeros_like(dfa), dfa)
                    l_dffa_d = BCE(tf.zeros_like(dffa), dffa)

                    l_dri_d = BCE(tf.ones_like(dri), dri)
                    l_dfi_d = BCE(tf.zeros_like(dfi), dfi)
                    l_dffi_d = BCE(tf.zeros_like(dffi), dffi)

                    l_drm_d = BCE(tf.ones_like(drm), drm)
                    l_dfm_d = BCE(tf.zeros_like(dfm), dfm)
                    l_dffm1_d = BCE(tf.zeros_like(dffm1), dffm1)
                    l_dffm2_d = BCE(tf.zeros_like(dffm2), dffm2)

                    # Simple and Cycle Consistency Loss
                    l__xa = MAE(ya, _xa)
                    l_xa = MAE(ya, xa)

                    l__xi = MAE(yi, _xi)
                    l_xi = MAE(yi, xi)

                    l__xm = MAE(ym, _xm)
                    l__xxm = MAE(ym, _xxm)
                    l_xm = MAE(ym, xm)

                    # Aggregate Generator Losses
                    ag_loss = tf.concat(
                        (
                            100
                            * (
                                l__xa * 0.5625
                                + l_xa * 0.34375
                                + l__xxm * 0.0625
                                + l_xm * 0.03125
                            ),
                            (
                                l_dfa_g * 0.5625
                                + l_dffa_g * 0.34375
                                + l_dffm1_g * 0.0625
                                + l_dffm2_g * 0.03125
                            ),
                        ),
                        axis=1,
                    )

                    ig_loss = tf.concat(
                        (
                            100
                            * (
                                l__xi * 0.5625
                                + l_xi * 0.34375
                                + l__xxm * 0.0625
                                + l_xm * 0.03125
                            ),
                            (
                                l_dfi_g * 0.5625
                                + l_dffi_g * 0.34375
                                + l_dffm1_g * 0.0625
                                + l_dffm2_g * 0.03125
                            ),
                        ),
                        axis=1,
                    )

                    mg_loss = tf.concat(
                        (
                            100 * (l__xm * 0.4375 + l__xxm * 0.375 + l_xm * 0.1875),
                            (l_dfm_g * 0.4375 + l_dffm1_g * 0.375 + l_dffm2_g * 0.1875),
                        ),
                        axis=1,
                    )

                    # Aggregate Discriminator Losses
                    ad_loss = l_dra_d * 0.5 + l_dfa_d * 0.375 + l_dffa_d * 0.125

                    id_loss = l_dri_d * 0.5 + l_dfi_d * 0.375 + l_dffi_d * 0.125

                    md_loss = (
                        l_drm_d * 0.5
                        + l_dfm_d * 0.40625
                        + l_dffm1_d * 0.0625
                        + l_dffm2_d * 0.03125
                    )

                    ag_pred = tf.concat(((_xa + xa) / 2, (dfa + dffa) / 2), axis=1)
                    ag_true = tf.concat((ya, tf.ones_like(dfa)), axis=1)
                    ig_pred = tf.concat(((_xi + xi) / 2, (dfi + dffi) / 2), axis=1)
                    ig_true = tf.concat((yi, tf.ones_like(dfi)), axis=1)
                    mg_pred = tf.concat(
                        ((_xm + _xxm + xm) / 3, (dfm + dffm1 + dffm2) / 3), axis=1
                    )
                    mg_true = tf.concat((ym, tf.ones_like(dfm)), axis=1)
                    ad_pred = tf.concat((dra, (dfa + dffa) / 2), axis=1)
                    ad_true = tf.concat((tf.ones_like(dra), tf.zeros_like(dfa)), axis=1)
                    id_pred = tf.concat((dri, (dfi + dffi) / 2), axis=1)
                    id_true = tf.concat((tf.ones_like(dri), tf.zeros_like(dfi)), axis=1)
                    md_pred = tf.concat((drm, (dfm + dffm1 + dffm2) / 3), axis=1)
                    md_true = tf.concat((tf.ones_like(drm), tf.zeros_like(dfm)), axis=1)

                    # Divide by batch_size because 'reduction' in losses
                    # is None (due to use inside tf.distribute.Strategy)
                    return dict(
                        ag_loss=tf.nn.compute_average_loss(
                            ag_loss, global_batch_size=self.batch_size
                        ),
                        ad_loss=tf.nn.compute_average_loss(
                            ad_loss, global_batch_size=self.batch_size
                        ),
                        ig_loss=tf.nn.compute_average_loss(
                            ig_loss, global_batch_size=self.batch_size
                        ),
                        id_loss=tf.nn.compute_average_loss(
                            id_loss, global_batch_size=self.batch_size
                        ),
                        mg_loss=tf.nn.compute_average_loss(
                            mg_loss, global_batch_size=self.batch_size
                        ),
                        md_loss=tf.nn.compute_average_loss(
                            md_loss, global_batch_size=self.batch_size
                        ),
                        ag_acc=(ag_true, ag_pred),
                        ad_acc=(ad_true, ad_pred),
                        ig_acc=(ig_true, ig_pred),
                        id_acc=(id_true, id_pred),
                        mg_acc=(mg_true, mg_pred),
                        md_acc=(md_true, md_pred),
                    )

                ya, yi, ym = inputs
                with tf.GradientTape(persistent=True) as tape:
                    meteric = block(ya, yi, ym)

                self.optimizer["ag_opt"].apply_gradients(
                    zip(
                        tape.gradient(
                            meteric["ag_loss"], acapella.generator.trainable_variables
                        ),
                        acapella.generator.trainable_variables,
                    )
                )
                self.optimizer["ad_opt"].apply_gradients(
                    zip(
                        tape.gradient(
                            meteric["ad_loss"],
                            acapella.discriminator.trainable_variables,
                        ),
                        acapella.discriminator.trainable_variables,
                    )
                )

                self.optimizer["ig_opt"].apply_gradients(
                    zip(
                        tape.gradient(
                            meteric["ig_loss"],
                            instrumental.generator.trainable_variables,
                        ),
                        instrumental.generator.trainable_variables,
                    )
                )
                self.optimizer["id_opt"].apply_gradients(
                    zip(
                        tape.gradient(
                            meteric["id_loss"],
                            instrumental.discriminator.trainable_variables,
                        ),
                        instrumental.discriminator.trainable_variables,
                    )
                )

                self.optimizer["mg_opt"].apply_gradients(
                    zip(
                        tape.gradient(
                            meteric["mg_loss"], mixer.generator.trainable_variables
                        ),
                        mixer.generator.trainable_variables,
                    )
                )
                self.optimizer["md_opt"].apply_gradients(
                    zip(
                        tape.gradient(
                            meteric["md_loss"], mixer.discriminator.trainable_variables
                        ),
                        mixer.discriminator.trainable_variables,
                    )
                )
                self.metrics["ag_loss"].update_state(meteric["ag_loss"])
                self.metrics["ig_loss"].update_state(meteric["ig_loss"])
                self.metrics["mg_loss"].update_state(meteric["mg_loss"])
                self.metrics["ad_loss"].update_state(meteric["ad_loss"])
                self.metrics["id_loss"].update_state(meteric["id_loss"])
                self.metrics["md_loss"].update_state(meteric["md_loss"])
                self.metrics["ag_acc"].update_state(*meteric["ag_acc"])
                self.metrics["ad_acc"].update_state(*meteric["ad_acc"])
                self.metrics["ig_acc"].update_state(*meteric["ig_acc"])
                self.metrics["id_acc"].update_state(*meteric["id_acc"])
                self.metrics["mg_acc"].update_state(*meteric["mg_acc"])
                self.metrics["md_acc"].update_state(*meteric["md_acc"])

            self.strategy.run(_step, args=(next(data_iterator)))

        for epoch in range(self.epochs):
            if self.verbose > 0:
                print(f"Epoch: {epoch}/{self.epochs}")
            steps = range(self.steps_per_epoch)
            steps = (
                tqdm(
                    steps,
                    bar_format="{n_fmt}/{total_fmt} |{bar}| {elapsed} {rate_inv_fmt}",
                )
                if self.verbose == 1
                else steps
            )
            data = iter(self.dataset)
            for _ in steps:
                train_step(data)
                if self.verbose > 0:
                    print(
                        f"Model\tGen Loss  \tGen Acc   \tDisc Loss \tDisc Acc  \n"
                        f"Acap.\t{self.metrics['ag_loss'].result(): <10.5f}\t"
                        f"{self.metrics['ag_acc'].result()*100: <10.5f}\t"
                        f"{self.metrics['ad_loss'].result(): <10.5f}\t"
                        f"{self.metrics['ad_acc'].result()*100: <10.5f}\n"
                        f"Instr\t{self.metrics['ig_loss'].result(): <10.5f}\t"
                        f"{self.metrics['ig_acc'].result()*100: <10.5f}\t"
                        f"{self.metrics['id_loss'].result(): <10.5f}\t"
                        f"{self.metrics['id_acc'].result()*100: <10.5f}\n"
                        f"Mixer\t{self.metrics['mg_loss'].result(): <10.5f}\t"
                        f"{self.metrics['mg_acc'].result()*100: <10.5f}\t"
                        f"{self.metrics['md_loss'].result(): <10.5f}\t"
                        f"{self.metrics['md_acc'].result()*100: <10.5f}\n"
                    )
            self.checkpoint.save(CONFIG.checkpoint_dir)
            with summary_writer.as_default():
                tf.summary.scalar(
                    "Acapella/Generator_loss",
                    self.metrics["ag_loss"].result(),
                    step=epoch,
                )
                tf.summary.scalar(
                    "Acapella/Discriminator_loss",
                    self.metrics["ad_loss"].result(),
                    step=epoch,
                )
                tf.summary.scalar(
                    "Instrumental/Generator_loss",
                    self.metrics["ig_loss"].result(),
                    step=epoch,
                )
                tf.summary.scalar(
                    "Instrumental/Discriminator_loss",
                    self.metrics["id_loss"].result(),
                    step=epoch,
                )
                tf.summary.scalar(
                    "Mixer/Generator_loss", self.metrics["mg_loss"].result(), step=epoch
                )
                tf.summary.scalar(
                    "Mixer/Discriminator_loss",
                    self.metrics["md_loss"].result(),
                    step=epoch,
                )
                tf.summary.scalar(
                    "Acapella/Generator_acc",
                    self.metrics["ag_acc"].result() * 100,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Acapella/Discriminator_acc",
                    self.metrics["ad_acc"].result() * 100,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Instrumental/Generator_acc",
                    self.metrics["ig_acc"].result() * 100,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Instrumental/Discriminator_acc",
                    self.metrics["id_acc"].result() * 100,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Mixer/Generator_acc",
                    self.metrics["mg_acc"].result() * 100,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Mixer/Discriminator_acc",
                    self.metrics["md_acc"].result() * 100,
                    step=epoch,
                )
            self.metrics["ag_loss"].reset_states()
            self.metrics["ig_loss"].reset_states()
            self.metrics["mg_loss"].reset_states()
            self.metrics["ad_loss"].reset_states()
            self.metrics["id_loss"].reset_states()
            self.metrics["md_loss"].reset_states()
            self.metrics["ag_acc"].reset_states()
            self.metrics["ig_acc"].reset_states()
            self.metrics["mg_acc"].reset_states()
            self.metrics["ad_acc"].reset_states()
            self.metrics["id_acc"].reset_states()
            self.metrics["md_acc"].reset_states()
