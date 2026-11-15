"""Draft anchored patch for Transformers 4.31 Trainer's training loop.

This code is not a training entry. It is intentionally fail-closed on source
drift and must pass real cross-epoch Trainer/DeepSpeed acceptance before use.
"""
import hashlib
import inspect
import textwrap

TRAINER_FILE_SHA256 = '24a579e7af77c3699020ac16c1b4ad16d64aa584dafe52a90d0807c381174f31'


def patch_text(source):
    source = textwrap.dedent(source)
    def change(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Trainer source anchor is absent or ambiguous: ' + old[:100])
        source = source.replace(old, new, 1)

    change('''            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )''', '''            num_train_epochs = self._beauty_epochs
            assert max_steps * args.gradient_accumulation_steps == num_train_epochs * len_dataloader''')
    change('''        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0''', '''        assert self._beauty_resume_state is not None
        epochs_trained = self._beauty_resume_state['loader_state']['resume_epoch']
        assert 0 <= epochs_trained < num_train_epochs
        steps_trained_in_current_epoch = 0''')
    change('''    model.zero_grad()

    self.control = self.callback_handler.on_train_begin''', '''    model.zero_grad()
    if resume_from_checkpoint is not None:
        self._restore_beauty_state_after_zero_grad()

    self.control = self.callback_handler.on_train_begin''')
    change('''    # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    if not args.ignore_data_skip:''', '''    # ReplayLoader restores sampler state; do not consume artificial epoch prefixes.
    if not args.ignore_data_skip and resume_from_checkpoint is None:''')
    change('''    total_batched_samples = 0
    for epoch in range(epochs_trained, num_train_epochs):''', '''    total_batched_samples = self._beauty_resume_state['micro_steps'] if resume_from_checkpoint else 0
    for epoch in range(epochs_trained, num_train_epochs):''')
    change('''        rng_to_sync = False
        steps_skipped = 0''', '''        rng_to_sync = False
        steps_skipped = 0
        if resume_from_checkpoint and epoch == epochs_trained:
            saved_yielded = self._beauty_resume_state['loader_state']['yielded']
            if saved_yielded < len_dataloader:
                steps_skipped = saved_yielded''')
    change('''            if step % args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin''', '''            if (total_batched_samples - 1) % args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin''')
    change('''        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        self._maybe_log_save_evaluate''', '''        assert total_batched_samples == (epoch + 1) * len_dataloader, 'Epoch ended before all microbatches were consumed'
        self.state.epoch = float(epoch + 1)
        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        self._maybe_log_save_evaluate''')
    change('''    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:''', '''    self._beauty_capture_pre_best()
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:''')
    compile(source, '<beauty_trainer_inner_loop>', 'exec')
    return source


def build(trainer_module):
    path = inspect.getsourcefile(trainer_module.Trainer)
    with open(path, 'rb') as stream:
        actual = hashlib.sha256(stream.read()).hexdigest()
    if actual != TRAINER_FILE_SHA256:
        raise ValueError('Unexpected Trainer file identity: ' + actual)
    source = patch_text(inspect.getsource(trainer_module.Trainer._inner_training_loop))
    namespace = dict(vars(trainer_module))
    exec(compile(source, '<beauty_trainer_inner_loop>', 'exec'), namespace)
    return namespace['_inner_training_loop'], hashlib.sha256(source.encode()).hexdigest()
