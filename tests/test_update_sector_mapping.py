import json

import pytest

from webui import update_sector_mapping


def load_real_vocabulary():
    return update_sector_mapping.load_vocabulary(
        update_sector_mapping.DEFAULT_VOCABULARY_PATH
    )


def test_build_mapping_preserves_ids_and_marks_new_labels_unknown():
    payload = update_sector_mapping.build_mapping_payload([
        ['2026-08-28', 'sh.600000', '浦发银行', 'J66货币金融服务', '证监会行业分类'],
        ['2026-08-28', 'sz.000001', '平安银行', '未来新增分类', '证监会行业分类'],
        ['2026-08-28', 'bj.430001', '北交所股票', 'C34通用设备制造业', '证监会行业分类'],
        ['invalid-date', 'sh.600004', '白云机场', 'G56航空运输业', '证监会行业分类'],
    ], load_real_vocabulary())

    assert payload['symbol_count'] == 2
    assert payload['unknown_sector_count'] == 1
    assert payload['symbols']['sh.600000']['sector_id'] == 63
    assert payload['symbols']['sz.000001']['sector_id'] == 86
    assert payload['reference_date'] == '2026-08-28'


def test_replacement_guard_rejects_unexpected_provider_shrink(tmp_path):
    output = tmp_path / 'mapping.json'
    output.write_text(json.dumps({
        'symbols': {f'sh.{index:06d}': {} for index in range(100)},
    }), encoding='utf-8')
    payload = {'symbols': {f'sh.{index:06d}': {} for index in range(50)}}

    with pytest.raises(ValueError, match='Refusing to shrink'):
        update_sector_mapping.validate_replacement(
            payload,
            output,
            minimum_symbols=1,
            minimum_retention=0.8,
            force=False,
        )


def test_atomic_writer_keeps_one_backup(tmp_path):
    output = tmp_path / 'mapping.json'
    output.write_text('{"version":"old"}\n', encoding='utf-8')

    update_sector_mapping.write_json_atomic(output, {'version': 'new'})

    assert json.loads(output.read_text(encoding='utf-8')) == {'version': 'new'}
    assert json.loads(
        (tmp_path / 'mapping.json.bak').read_text(encoding='utf-8')
    ) == {'version': 'old'}
