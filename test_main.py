import pathlib
import tempfile

import unittest
import argparse
import main
import os


def get_base_args():
  args = [
      '-b', '2', '--opt-version', 'facebook/opt-125m',
      '--val-steps-per-epoch', '2', '--epochs', '1', '--steps-per-epoch', '2',
      '--text-emb-layers', '-1', '--shared-emb-dim', '256',
      '--n-visual-tokens', '1', '--visual-model', 'openai/clip-vit-base-patch32', '--concat-captions-prob', '0.5']
  return args

def check_workdir_outputs(workdir_path):
  workdir_content = os.listdir(workdir_path)
  print('workdir content: %s', workdir_content)

  assert 'ckpt.pth.tar' in workdir_content
  assert 'model_args.json' in workdir_content
  assert 'param_count.txt' in workdir_content
  assert 'git_info.txt' in workdir_content
  assert any(['events.out.tfevents' in fn for fn in workdir_content])


class MultitaskTrainTest(unittest.TestCase):
  """Test captioning."""
  def test_train_and_evaluate(self):
    workdir = tempfile.mkdtemp()
    proj_root_dir = pathlib.Path(__file__).parents[0]
    exp_name = 'test_multitask'

    parser = argparse.ArgumentParser(description='Unit test parser')
    args = get_base_args() + ['--log-base-dir', workdir, '--exp_name', exp_name]
    main.main(args)
    check_workdir_outputs(os.path.join(workdir, exp_name))


if __name__ == '__main__':
  unittest.main()
