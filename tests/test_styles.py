import json
from pathlib import Path
import tempfile
import unittest
from composer.styles import REGISTRY, STYLE_ORDER, StyleRegistry, get_style


class StyleRegistryTests(unittest.TestCase):
    def test_all_twelve_builtin_styles_load_and_validate(self):
        expected = {
            'ambient', 'house', 'lofi', 'cinematic',
            'synthwave', 'latin', 'funk', 'dnb',
            'techno', 'jazz', 'afrobeat', 'pop',
        }
        self.assertTrue(expected.issubset(set(STYLE_ORDER)), f'Missing styles: {expected - set(STYLE_ORDER)}')
        self.assertGreaterEqual(len(STYLE_ORDER), 12)
        for key in expected:
            style = get_style(key)
            self.assertEqual(style.key, key)
            self.assertEqual(len(style.presets), 4)
            self.assertGreaterEqual(len(style.progression), 4)
            self.assertGreater(len(style.motif_a), 2)
            self.assertGreater(len(style.motif_b), 2)
            self.assertGreater(len(style.bass_pattern), 1)
            self.assertGreater(len(style.drums.fill_bar4), 1)
            self.assertLessEqual(style.bpm_min, style.bpm_default)
            self.assertLessEqual(style.bpm_default, style.bpm_max)

    def test_navigation_cycles_cleanly(self):
        first = STYLE_ORDER[0]
        second = REGISTRY.next_key(first, 1)
        self.assertEqual(second, STYLE_ORDER[1])
        self.assertEqual(REGISTRY.next_key(first, -1), STYLE_ORDER[-1])

    def test_custom_json_style_discovered_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            preset_path = Path(tmp) / 'custom_test.json'
            # Clone ambient preset into temp folder under a new key.
            src = Path(__file__).resolve().parents[1] / 'composer/styles/presets/ambient.json'
            data = json.loads(src.read_text(encoding='utf-8'))
            data['key'] = 'custom_test'
            data['label'] = 'Custom Test Style'
            preset_path.write_text(json.dumps(data), encoding='utf-8')

            reg = StyleRegistry(Path(tmp))
            self.assertEqual(reg.keys, ('custom_test',))
            self.assertEqual(reg.get('custom_test').label, 'Custom Test Style')


if __name__ == '__main__':
    unittest.main()
