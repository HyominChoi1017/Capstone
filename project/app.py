from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.nn as nn
from performer_pytorch import Performer
import torch.nn.functional as F 
import tracemalloc, time

app = Flask(__name__)

decoding_dict = {
    -1:'',
    0: '타다',
    1: '연착',
    2: '필요없다',
    3: '표지판',
    4: '끝',
    5: '있다',
    6: '택시',
    7: '다르다',
    8: '짐',
    9: '화장실',
    10: '샛길',
    11: '가방',
    12: '돈주다',
    13: '손바닥찍다',
    14: '아이',
    15: '이마트',
    16: '도움받다',
    17: '위험',
    18: '첫차',
    19: '용산역',
    20: '배',
    21: '다시',
    22: '계산',
    23: '의자',
    24: '돈받다',
    25: '사람',
    26: '힘들다',
    27: '조심',
    28: '회의실',
    29: '확인',
    30: '이화여대',
    31: '학교',
    32: '파리바게트',
    33: '분당',
    34: '잠깐',
    35: '자판기',
    36: '육교',
    37: '천호',
    38: '반지',
    39: '불편하다',
    40: '아니다',
    41: '돈얼마',
    42: '시청',
    43: '실수하다',
    44: '하나',
    45: '내리다',
    46: '119',
    47: '장애인복지카드',
    48: '공기청정기',
    49: '청음회관',
    50: '따뜻하다',
    51: '지금',
    52: '엘리베이터',
    53: '올리다',
    54: '왼쪽',
    55: '안녕하세요',
    56: '충분하다',
    57: '사거리',
    58: '계단',
    59: '난방',
    60: '잠실대교',
    61: '지갑',
    62: '공항',
    63: '수고',
    64: '서울역',
    65: '뒤',
    66: '물품보관',
    67: '만나다',
    68: '응급실',
    69: '송파',
    70: '기차',
    71: '대략',
    72: '롯데월드',
    73: '어지럽다',
    74: '천안아산역',
    75: '스타벅스',
    76: '운전',
    77: '편의점',
    78: '항상',
    79: '보건소',
    80: '안되다',
    81: '발생하다',
    82: '약',
    83: '경찰',
    84: '신분증',
    85: '오른쪽',
    86: '종로',
    87: '말해주다',
    88: '어린이교통카드',
    89: '도착',
    90: '서울농아인협회',
    91: '좌회전',
    92: '오천원',
    93: '사진기',
    94: '좋다',
    95: '유턴',
    96: '맞다',
    97: '표',
    98: '버스',
    99: '춥다',
    100: '건너다',
    101: '없다',
    102: '1호',
    103: '곳곳',
    104: '우회전',
    105: '신분당',
    106: '차',
    107: '지하철',
    108: '오늘',
    109: '은행',
    110: '누구',
    111: '보청기',
    112: '카드',
    113: '한국농아인협회',
    114: '꼿다',
    115: '단말기터치',
    116: '나',
    117: '시간',
    118: '맥도날드',
    119: '8호',
    120: '안내소',
    121: '차내리다',
    122: '고속',
    123: '2호',
    124: '길',
    125: '3',
    126: '언덕',
    127: '차따라가다',
    128: '모르다',
    129: '핸드폰',
    130: '만원',
    131: '여의도',
    132: '역무원',
    133: '50분',
    134: '이름',
    135: '밤',
    136: '4사람',
    137: '가능',
    138: '우산',
    139: '고속터미널',
    140: '대로',
    141: '지연되다',
    142: '신호등',
    143: '서울대학교',
    144: '자전거',
    145: '병원',
    146: '다음',
    147: '차내리다\n',
    148: '트렁크열다',
    149: '명동',
    150: '괜찮다',
    151: '지하철오다',
    152: '안경',
    153: '9호',
    154: '당신',
    155: '교통카드',
    156: '어떻게',
    157: '접근',
    158: '지름길',
    159: '왜',
    160: '몇사람',
    161: '원래',
    162: '국립박물관',
    163: '빨리',
    164: '막차',
    165: '안전벨트',
    166: '서울',
    167: '약수',
    168: '서대문농아인복지관',
    169: '그남자',
    170: '감사합니다',
    171: '가다',
    172: '에어컨',
    173: '열쇠',
    174: '시계',
    175: '편지',
    176: '알다',
    177: '한달',
    178: '나사렛',
    179: '백화점',
    180: '갈아타다',
    181: '약속',
    182: '필요',
    183: '여기',
    184: '미리',
    185: '5호',
    186: '저기',
    187: '마포대교',
    188: '무엇',
    189: '유리',
    190: '미안합니다',
    191: '차밀리다',
    192: '노트북',
    193: '곳',
    194: '말하다',
    195: '군청',
    196: '어린이집',
    197: '강남',
    198: '영수증',
    199: '엉덩이',
    200: '물건',
    201: '급하다',
    202: '화나다',
    203: '아파트',
    204: '영등포',
    205: '터널',
    206: '돈',
    207: '설명'
}

class PerformerClassifier(nn.Module):
        def __init__(self, input_dim, model_dim, num_classes):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, model_dim)
            self.performer = Performer(
                dim=model_dim,
                dim_head=32,
                depth=6,
                heads=8,
                causal=False
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, num_classes)
            )

        def forward(self, x, mask):
            # x: (B, T, D)
            x = self.input_proj(x)  # (B, T, model_dim)
            x = self.performer(x, mask=mask)  # (B, T, model_dim)
            x = x.mean(dim=1)  # 간단한 average pooling
            return self.classifier(x)  # (B, num_classes)
        
model = PerformerClassifier(input_dim=336, model_dim=256, num_classes=208)


@app.route('/donotsleep', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "message": "Wake up!"})

@app.route('/', methods=['GET'])
def test():
    return jsonify({"message": "반갑다. 잘 돌아간다."})

# POST 요청: JSON Body 받기
@app.route('/ai', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        print("/ai에서 요청을 받았습니다.")

        print("data:", data )

        tracemalloc.start()  # 메모리 추적 시작
        data_np = np.array(data['data'], dtype=np.float32) 
        current, peak = tracemalloc.get_traced_memory()
        print(f"현재 메모리: {current / 1024**2:.2f} MB")
        print(f"최대 메모리: {peak / 1024**2:.2f} MB")

        prev_result = data['prev_result']
        print("data_np shape:", data_np.shape)
        print("prev_result:", prev_result)
        
        video_np = (data_np - np.min(data_np)) / (np.max(data_np) - np.min(data_np) + 1e-6)
        
        print("tensor로 바꾸고 masking 진행")
        tracemalloc.stop()  # 메모리 추적 중지
        tracemalloc.start()  # 메모리 추적 다시 시작
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_data = torch.from_numpy(video_np).unsqueeze(0).float().to(device)  # (1, L, 336)
        mask = torch.from_numpy(video_np).unsqueeze(0).bool().to(device)          # (1, L)
        
        
        
        current, peak = tracemalloc.get_traced_memory()
        
        print(f"현재 메모리: {current / 1024**2:.2f} MB")
        print(f"최대 메모리: {peak / 1024**2:.2f} MB")

        tracemalloc.stop()  # 메모리 추적 중지
        tracemalloc.start()  # 메모리 추적 다시 시작
        start = time.time()
        with torch.no_grad():
            print("모델에 입력하고 예측 진행")
            logits = model(torch_data, mask)
        end = time.time()
        print(f"모델 예측 시간: {end - start:.2f}초")
        probs = F.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=1)
        threshold = 0.5
        final_preds = torch.where(max_probs > threshold, preds, torch.tensor(-1))

        predicted_class = int(final_preds.cpu())
        predicted_word = decoding_dict.get(predicted_class, '')
        current, peak = tracemalloc.get_traced_memory()
        print(f"현재 메모리: {current / 1024**2:.2f} MB")
        print(f"최대 메모리: {peak / 1024**2:.2f} MB")

        if predicted_word != prev_result:
            return jsonify({"you_sent": predicted_word})
        else:
            return jsonify({"you sent":""})  # 변화 없으면 빈 응답
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 여기 필요한 변수들을 초기화하면 되건가?

    model.load_state_dict(torch.load("project/full_model.pth", map_location='cpu'))
    model.eval()

    

    # Flask 서버 실행
    app.run(debug=True)
