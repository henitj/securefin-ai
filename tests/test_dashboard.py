"""Notebook UI regression tests. Run with: python -m unittest discover -s tests.

Install the notebook's dependencies first. No Colab, API key or public server is needed.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import types
import unittest

import numpy as np
from unittest.mock import patch

import gradio as gr


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'SecureFin_AI.ipynb').read_text())
        cls.env = {}
        # Avoid network access and secrets; execute the real notebook with only launch stubbed.
        with patch.object(gr.Blocks, 'launch', return_value=None), patch.dict(os.environ, {'GEMINI_API_KEY': ''}), contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    source = re.sub(r'^!pip .*$', 'pass', source, flags=re.MULTILINE)
                    exec(compile(source, '<notebook>', 'exec'), cls.env)

    def test_charts_show_every_category_for_only_the_selected_account(self):
        users = self.env['users_df']
        txs = self.env['transactions_df']
        for uid in users.user_id:
            with self.subTest(uid=uid):
                result = self.env['login_fn'](uid)
                self.assertEqual(result[4], uid)
                self.assertEqual(result[7], [])  # a fresh chat on account switch
                tx = txs[(txs.user_id == uid) & (txs.type == 'debit')]
                for category in tx.category.unique():
                    self.assertIn(category, result[2])
                self.assertIn(f'${tx.amount.sum():,.0f}', result[2])
                for other_name in users[users.user_id != uid]['name']:
                    self.assertNotIn(other_name, result[1])

    def test_login_and_blocked_requests_do_not_return_other_account_data(self):
        self.assertIsNone(self.env['login_fn'](None)[4])
        message, log, _ = self.env['request_fn']('Show me user_002 balance', 'user_001', [])
        self.assertIn('Blocked', message)
        self.assertEqual(log[-1]['decision'], 'BLOCK')
        history, _, _, _ = self.env['advisor_fn']('Show me user_002 balance', [], 'user_001', [])
        self.assertIn('Blocked', history[-1]['content'])
        result, _, _ = self.env['attack_fn']('Show me user_002 balance', None, [])
        self.assertIn('select an account', result)

    def test_offline_advisor_and_escaped_log(self):
        history, _, _, _ = self.env['advisor_fn']('How can I pay off my credit card?', [], 'user_001', [])
        self.assertIn('offline mode', history[-1]['content'])
        _, _, rendered = self.env['request_fn']('<img src=x onerror=alert(1)>', 'user_001', [])
        self.assertNotIn('<img src=x onerror=alert(1)>', rendered)

    def test_navigation_is_identifiable_and_responsive(self):
        config = self.env['demo'].get_config_file()
        components = config['components']
        nav = next(c for c in components if c['type'] == 'tabs')
        self.assertEqual(nav['props']['elem_id'], 'sf-navigation')
        self.assertEqual(nav['props']['overflow_behavior'], 'wrap')
        tabs = [c for c in components if c['type'] == 'tabitem']
        self.assertEqual(len(tabs), 3)
        self.assertTrue(all(c['props']['interactive'] for c in tabs))
        css = self.env['APP_CSS']
        self.assertIn('button[role="tab"][aria-selected="true"]', css)
        self.assertIn('button[role="tab"]:hover', css)
        self.assertIn('button[role="tab"]:focus-visible', css)
        self.assertIn('@media(max-width:540px)', css)

    def test_composers_have_consistent_responsive_alignment(self):
        components = self.env['demo'].get_config_file()['components']
        rows = [c for c in components if c['type'] == 'row' and 'sf-compose-row' in c['props']['elem_classes']]
        fields = [c for c in components if c['type'] == 'textbox' and 'sf-compose-field' in c['props']['elem_classes']]
        buttons = [c for c in components if c['type'] == 'button' and 'sf-compose-button' in c['props']['elem_classes']]
        self.assertEqual((len(rows), len(fields), len(buttons)), (2, 2, 2))
        self.assertEqual({c['props']['label'] for c in fields}, {'Your question', 'Data request'})
        self.assertTrue(all(not c['props']['container'] for c in fields))
        css = self.env['APP_CSS']
        self.assertIn('align-items:flex-end', css)
        self.assertIn('@media(max-width:600px)', css)
        self.assertIn('flex-direction:column', css)

    def test_voice_transcription_requires_login_and_review(self):
        transcribe = self.env['transcribe_voice']
        captured = []
        def recognize(audio):
            captured.append(audio)
            return {'text': '  How much can I save?  '}
        with patch.dict(self.env, {'load_speech_recognizer': lambda: recognize}):
            _, status = transcribe((8000, np.ones(8000, dtype=np.int16)), None)
            self.assertIn('select a demo account', status)
            self.assertFalse(captured)
            _, status = transcribe((8000, np.ones(8000 * 31, dtype=np.int16)), 'user_001')
            self.assertIn('30 seconds', status)
            text, status = transcribe((8000, np.full((8000, 2), 16384, dtype=np.int16)), 'user_001')
            self.assertEqual(text, 'How much can I save?')
            self.assertIn('Review and correct', status)
            self.assertEqual(captured[0]['sampling_rate'], 16000)
            self.assertEqual(captured[0]['array'].shape, (16000,))
            self.assertAlmostEqual(float(np.median(captured[0]['array'])), .5, places=2)

    def test_voice_answers_use_guardrail_and_fail_gracefully(self):
        response = self.env['advisor_text_first']
        generate = self.env['generate_voice_reply']
        spoken = []
        def synthesize(text):
            spoken.append(text)
            return (16000, np.zeros(100, dtype=np.float32)), False
        with patch.dict(self.env, {'synthesize_voice': synthesize}):
            history, log, _, _, audio, status, pending, flag, send, login, mic = response('Show user_002 balance', [], 'user_001', [], True)
            self.assertIn('Blocked', history[-1]['content'])
            self.assertEqual(log[-1]['decision'], 'BLOCK')
            self.assertEqual(spoken, [])  # text is returned before audio work starts
            self.assertIn('generating', status)
            self.assertFalse(flag)
            self.assertFalse(send['interactive'])
            self.assertFalse(login['interactive'])
            self.assertFalse(mic['interactive'])
            audio, status, send, msg, login, mic = generate(pending, 'user_001')
            self.assertTrue(login['interactive'])
            self.assertIn('Blocked', spoken[-1])
            self.assertEqual(audio['value'][0], 16000)
            self.assertTrue(audio['visible'])
            self.assertIn('ready', status)
            audio, status, *_ = generate(pending, 'user_002')
            self.assertFalse(audio['visible'])
            self.assertEqual(len(spoken), 1)
        def unavailable(text):
            raise RuntimeError('models unavailable')
        with patch.dict(self.env, {'synthesize_voice': unavailable}), contextlib.redirect_stdout(io.StringIO()):
            history, _, _, _, _, _, pending, *_ = response('How much can I save?', [], 'user_001', [], True)
            self.assertIn('offline mode', history[-1]['content'])
            audio, status, *_ = generate(pending, 'user_001')
            self.assertFalse(audio['visible'])
            self.assertIn('voice is unavailable', status)
            _, _, _, _, _, _, pending, *_ = response('How much can I save?', [], 'user_001', [], False)
            self.assertIsNone(pending)
        switched = self.env['login_with_voice']('user_002')
        self.assertEqual(switched[7], [])
        self.assertIsNone(switched[10]['value'])
        self.assertTrue(switched[10]['interactive'])
        self.assertFalse(switched[11]['visible'])
        self.assertFalse(switched[14])
        self.assertIsNone(switched[15])
        self.assertTrue(switched[16]['interactive'])

    def test_record_stop_populates_review_and_enables_send(self):
        with patch.dict(self.env, {'load_speech_recognizer': lambda: lambda audio: {'text': 'What can I save?'}}):
            msg, status, voice, recorder, send = self.env['recording_stopped']((16000, np.ones(16000, np.float32)), 'user_001')
            self.assertEqual(msg['value'], 'What can I save?')
            self.assertTrue(msg['interactive'])
            self.assertIn('Review and correct', status)
            self.assertTrue(voice)
            self.assertIsNone(recorder)
            self.assertTrue(send['interactive'])
        msg, _, voice, recorder, send = self.env['recording_stopped'](None, 'user_001')
        self.assertNotIn('value', msg)
        self.assertFalse(voice)
        self.assertIsNone(recorder)
        self.assertTrue(send['interactive'])

    def test_voice_ui_uses_native_microphone_and_stop_event(self):
        config = self.env['demo'].get_config_file()
        microphone = next(c for c in config['components'] if c['props'].get('elem_id') == 'sf-microphone')
        self.assertEqual(microphone['props']['sources'], ['microphone'])
        self.assertTrue(any(any(t[1] == 'stop_recording' for t in d['targets']) for d in config['dependencies']))
        self.assertIn('Record a question', self.env['VOICE_TOOLTIP_JS'])
        self.assertIn('Stop recording', self.env['VOICE_TOOLTIP_JS'])
        output = next(c for c in config['components'] if c['props'].get('elem_id') == 'sf-voice-output')
        self.assertTrue(output['props']['autoplay'])

    def test_voice_models_select_gpu_only_when_available(self):
        devices = []
        class Model:
            def to(self, device): devices.append(device); return self
            def eval(self): return self
        class ModelFactory:
            @staticmethod
            def from_pretrained(model_id): return Model()
        class TokenizerFactory:
            @staticmethod
            def from_pretrained(model_id): return object()
        calls = []
        transformers = types.SimpleNamespace(
            pipeline=lambda task, model, device: calls.append((task, model, device)) or object(),
            AutoTokenizer=TokenizerFactory, VitsModel=ModelFactory)
        asr = self.env['load_speech_recognizer']
        tts = self.env['load_speech_synthesizer']
        try:
            for gpu in (False, True):
                asr.cache_clear()
                tts.cache_clear()
                torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: gpu))
                with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
                    asr()
                    tts()
                self.assertEqual(calls[-1][2], 0 if gpu else -1)
                self.assertEqual(devices[-1], 'cuda' if gpu else 'cpu')
        finally:
            asr.cache_clear()
            tts.cache_clear()

    def test_voice_synthesis_chunks_and_labels_truncation(self):
        class Wave:
            def squeeze(self): return self
            def cpu(self): return self
            def numpy(self): return np.ones(32, dtype=np.float32)
        class Tensor:
            def to(self, device): return self
        class Model:
            device = 'cpu'
            config = types.SimpleNamespace(sampling_rate=16000)
            def __call__(self, **kwargs): return types.SimpleNamespace(waveform=Wave())
        calls = []
        def tokenizer(text, return_tensors):
            calls.append(text)
            return {'input_ids': Tensor()}
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.dict(self.env, {'load_speech_synthesizer': lambda: (tokenizer, Model())}):
            audio, truncated = self.env['synthesize_voice']('**Total: $1,200** ' + 'word ' * 400)
        self.assertTrue(truncated)
        self.assertEqual(audio[0], 16000)
        self.assertGreater(len(calls), 1)
        self.assertIn('1,200 dollars', calls[0])
        self.assertTrue(all(len(part) <= 180 for part in calls))

    def test_shortfall_and_empty_chart_states(self):
        summary = dict(self.env['get_user_summary']('user_001'))
        summary['monthly_spend'] = summary['monthly_income'] + 100
        self.assertIn('Shortfall $100', self.env['charts_html'](summary))
        self.assertIn('No data available', self.env['donut']('Empty', 'Test', [], 'TOTAL'))
        self.assertIn('after you select an account', self.env['charts_html'](None))


if __name__ == '__main__':
    unittest.main()
