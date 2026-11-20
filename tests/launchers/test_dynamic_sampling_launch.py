import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

path = Path(__file__).resolve().parents[2] / 'evaluation/manage_dynamic_sampling_launch.py'
spec = importlib.util.spec_from_file_location('launch', path)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def call_main(args):
    with patch.object(sys, 'argv', ['launch', '--gpus', '1', *args]), \
         patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'RETRIEVER_URL': 'http://127.0.0.1:8000/retrieve'}):
        launch.main()


def test_check_only_reuses_retriever_and_never_trains():
    with patch.object(launch, 'healthy', return_value=True), \
         patch.object(launch.subprocess, 'run') as run, \
         patch.object(launch.subprocess, 'Popen') as popen:
        call_main(['--check-only'])
        popen.assert_not_called()
        commands = [x.args[0] for x in run.call_args_list]
        assert len(commands) == 3
        assert commands[0][-1] == '--print-config'
        assert '--skip-gpu-checks' in commands[1]
        assert Path(commands[2][1]).name == 'check_behavior_inputs.py'
        assert all('main_dynamic_sampling' not in ' '.join(x) for x in commands)


def test_unhealthy_occupied_port_refuses_launch():
    sock = MagicMock()
    sock.__enter__.return_value.connect_ex.return_value = 0
    with patch.object(launch, 'healthy', return_value=False), \
         patch.object(launch.socket, 'socket', return_value=sock), \
         patch.object(launch.subprocess, 'Popen') as popen:
        with pytest.raises(RuntimeError, match='occupied'):
            call_main(['--check-only'])
        popen.assert_not_called()


def test_external_unhealthy_url_does_not_start_local_service():
    with patch.object(sys, 'argv', ['launch', '--gpus', '1']), \
         patch.dict(os.environ, {'RETRIEVER_URL': 'http://remote:8000/retrieve'}), \
         patch.object(launch, 'healthy', return_value=False), \
         patch.object(launch.subprocess, 'Popen') as popen:
        with pytest.raises(RuntimeError, match='auto-start'):
            launch.main()
        popen.assert_not_called()


def test_normal_mode_executes_only_selected_trainer():
    with patch.object(launch, 'healthy', return_value=True), \
         patch.object(launch.subprocess, 'run') as run:
        call_main(['trainer.total_training_steps=10'])
        command = run.call_args.args[0]
        assert Path(command[1]).name == 'train_grpo_dynamic_sampling_1gpu.sh'
        assert command[2:] == ['trainer.total_training_steps=10']


def test_owned_service_stops_when_trainer_fails():
    sock = MagicMock()
    sock.__enter__.return_value.connect_ex.return_value = 1
    child = MagicMock()
    child.poll.return_value = None
    child.pid = 123456
    with patch.object(launch, 'healthy', side_effect=[False, True]), \
         patch.object(launch.socket, 'socket', return_value=sock), \
         patch.object(launch.subprocess, 'Popen', return_value=child), \
         patch.object(launch.subprocess, 'run', side_effect=RuntimeError('trainer failure')), \
         patch.object(launch.os, 'killpg') as kill:
        with pytest.raises(RuntimeError, match='trainer failure'):
            call_main([])
        kill.assert_called_once_with(child.pid, launch.signal.SIGTERM)
        child.wait.assert_called_once()
