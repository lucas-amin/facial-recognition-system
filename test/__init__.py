import base64
from unittest.mock import patch

import cv2


@pytest.fixture
def handler():
    handler = FacialRecognizer()
    return handler


def get_payload(filename):
    img = cv2.imread(filename)[:, :, ::-1]
    _, img_encoded = cv2.imencode('.jpg', img)
    image_data_encoded = base64.b64encode(img_encoded).decode()
    payload = {'image': image_data_encoded}
    return payload


@patch("src.data.qdrant_sponsor_handler.QdrantClient.scroll", autospec=True)
def test_list_all_sponsor_names(mock_get_sponsor_match, handler):
    mock_get_sponsor_match.return_value = ([ScoredPointMock(payload={'sponsor_name': 'A'}),
                                            ScoredPointMock(payload={'sponsor_name': 'B'}),
                                            ScoredPointMock(payload={'sponsor_name': 'C'}),
                                            ScoredPointMock(payload={'sponsor_name': 'D'})], None)

    league = "all"
    names_list = handler.get_sponsor_names(league, include_cannonical=True, rightsholder_id=None)
    assert isinstance(names_list, list)
    assert len(names_list) == 4
    assert isinstance(names_list[0], str)

