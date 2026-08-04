Vision Transformer (ViT) 완전 정리
1. 등장 배경

2017년 Transformer가 자연어처리(NLP)에서 폭발적 성공을 거둔 이후, "이 구조를 이미지에도 적용할 수 없을까?"라는 질문에서 시작된 모델이 ViT입니다. 2020년 구글 브레인 팀이 발표한 논문 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"에서 제안되었습니다.

그 전까지 컴퓨터 비전은 CNN(합성곱 신경망)이 지배적이었습니다. CNN은 지역적 특징(local feature)을 추출하는 convolution 연산과, 위치에 무관하게 같은 필터를 적용하는 translation invariance, 그리고 계층적으로 특징을 추상화하는 구조 덕분에 이미지 데이터에 잘 맞는 "귀납적 편향(inductive bias)"을 내재하고 있었습니다. ViT는 이런 이미지 특화 구조를 거의 사용하지 않고, 순수한 Transformer 구조만으로 이미지를 처리하려는 시도였다는 점에서 당시 파격적이었습니다.

2. 핵심 아이디어: "이미지를 단어처럼 취급하자"

NLP에서 Transformer는 문장을 토큰(단어) 시퀀스로 받아 처리합니다. ViT는 이미지를 작은 patch(조각)들로 잘라, 각 patch를 하나의 "토큰"처럼 취급합니다. 예를 들어 224x224 이미지를 16x16 크기의 patch로 나누면 14x14 = 196개의 patch(토큰)가 생기고, 이 196개의 시퀀스를 Transformer에 넣습니다.

즉, "이미지 하나 = 196개의 단어로 이루어진 문장"이라는 발상입니다.

3. 상세 작동 원리
(1) Patch Embedding (패치 분할 및 임베딩)
입력 이미지 (H x W x C)를 고정 크기 P x P의 patch들로 분할합니다.
각 patch를 1차원 벡터로 펼친 뒤(flatten), 학습 가능한 선형변환(linear projection)을 통해 D차원 임베딩 벡터로 변환합니다.
이 과정은 실제로는 커널 크기와 stride가 모두 P인 convolution 연산 한 번으로 구현됩니다(수학적으로 동일).
(2) [CLS] 토큰 추가
BERT에서 착안하여, 패치 임베딩들 앞에 학습 가능한 [CLS] 토큰(클래스 토큰)을 하나 붙입니다.
이 토큰은 Transformer를 거치면서 전체 이미지의 정보를 종합적으로 집약하는 역할을 하며, 최종적으로 이 토큰의 출력이 분류(classification)에 사용됩니다.
(3) Position Embedding (위치 정보 추가)
Transformer의 self-attention은 순서/위치 정보를 자체적으로 알지 못하는 구조(permutation invariant)이기 때문에, 각 patch가 이미지 내 어디에 위치했는지 알려주는 position embedding을 patch embedding에 더해줍니다.
ViT는 보통 학습 가능한(learnable) 1D position embedding을 사용합니다. (2D-aware embedding도 실험했지만 성능 차이는 크지 않았습니다.)
(4) Transformer Encoder

이렇게 만들어진 (196+1)개의 벡터 시퀀스가 표준 Transformer Encoder를 L번 통과합니다. 각 encoder block은 다음 구조입니다.

x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))

Multi-Head Self-Attention (MSA)
각 patch(토큰)가 다른 모든 patch와의 관계(유사도)를 계산하여, 어떤 patch에 더 주목할지 가중치를 부여합니다.

Query(Q), Key(K), Value(V) 세 벡터를 각 토큰에서 생성
Attention score = softmax(QK^T / √d_k)
이 score로 V를 가중합하여 새로운 표현을 얻음
여러 개의 head를 병렬로 두어(multi-head), 서로 다른 관점(부분 공간)에서 관계를 포착

이 attention 연산 덕분에 ViT는 patch 간의 거리와 상관없이 모든 patch가 서로 직접 상호작용할 수 있습니다. 이는 CNN이 convolution의 receptive field를 층층이 넓혀가며 간접적으로 전역 정보를 얻는 것과 근본적으로 다른 방식입니다. ViT는 첫 레이어부터 이미지 전체를 한 번에 바라볼 수 있는 global receptive field를 가집니다.

MLP (Feed-Forward Network)
각 토큰마다 독립적으로 적용되는 2층짜리 완전연결층(보통 GELU 활성화 함수 사용)으로, 비선형 변환을 통해 표현력을 높입니다.

LayerNorm & Residual Connection
학습 안정성을 위해 각 sub-layer 전에 LayerNorm을 적용(Pre-LN 방식)하고, residual connection(skip connection)으로 gradient 흐름을 원활하게 합니다.

(5) 분류 헤드 (Classification Head)
마지막 encoder layer를 통과한 [CLS] 토큰의 출력 벡터에 LayerNorm과 MLP(간단한 1~2층)를 적용하여 최종 클래스를 예측합니다.
4. 학습 방식의 특징

ViT의 성능은 데이터 규모에 극도로 민감합니다.

ImageNet(약 130만 장) 같은 중간 규모 데이터셋만으로 처음부터 학습(from scratch)하면, 비슷한 크기의 CNN(ResNet 등)보다 성능이 오히려 떨어집니다.
하지만 JFT-300M(3억 장), ImageNet-21k(1400만 장) 같은 초대형 데이터셋으로 사전학습(pretraining) 후 다운스트림 태스크에 fine-tuning하면, CNN을 능가하는 성능을 보입니다.

이는 CNN이 가진 convolution의 지역성, 가중치 공유, translation equivariance 같은 "이미지에 대한 사전 지식(inductive bias)"을 ViT는 갖고 있지 않기 때문입니다. ViT는 이런 사전 지식을 데이터로부터 스스로 학습해야 하므로, 학습 데이터가 충분해야만 그 유연함이 CNN의 구조적 이점을 앞지릅니다. 즉 "적은 가정 + 많은 데이터 = 더 나은 일반화"라는 Transformer 특유의 철학이 비전에서도 재현된 것입니다.

5. 장점
전역적 문맥 파악(Global Context): 첫 레이어부터 이미지 전체 patch 간 관계를 직접 모델링할 수 있어, 멀리 떨어진 부분들 간의 관계(예: 이미지 양 끝에 있는 물체 간 연관성)를 CNN보다 쉽게 포착합니다.
뛰어난 확장성(Scalability): 모델 크기와 데이터 크기를 키울수록 성능이 꾸준히 향상되는 경향(scaling law)이 CNN보다 뚜렷하게 나타납니다. 대규모 사전학습에 매우 유리한 구조입니다.
아키텍처 통일성: NLP와 비전이 동일한 Transformer 구조를 공유하게 되어, 멀티모달 모델(CLIP, Flamingo, GPT-4V류 등) 설계가 훨씬 수월해졌습니다.
유연한 입력 처리: patch 단위로 처리하므로 다양한 해상도나 형태에 비교적 유연하게 대응 가능하고, self-attention을 통해 masked autoencoding(MAE) 같은 self-supervised 학습법과도 궁합이 좋습니다.
해석 가능성: attention map을 시각화하면 모델이 이미지의 어느 부분에 주목하는지 비교적 직관적으로 확인할 수 있습니다.
6. 단점
막대한 데이터 요구량: 앞서 언급했듯 inductive bias가 약해 소규모 데이터셋에서는 CNN보다 성능이 떨어지거나 과적합되기 쉽습니다.
높은 연산/메모리 비용: Self-attention의 계산 복잡도는 토큰 개수 n에 대해 O(n²)입니다. 이미지 해상도가 커질수록 patch 개수가 급증하므로(예: 해상도를 2배로 하면 patch 수는 4배), 고해상도 이미지 처리 시 연산량과 메모리 사용량이 CNN보다 훨씬 빠르게 증가합니다.
지역적 세부정보 처리의 약점: patch를 독립적으로 선형 임베딩하기 때문에, patch 내부의 미세한 텍스처나 경계 정보를 CNN만큼 촘촘하게 잡아내지 못하는 경향이 있습니다(특히 저해상도 patch일 때).
작은 물체 탐지에 상대적으로 불리: object detection, segmentation처럼 정밀한 픽셀/지역 단위 정보가 중요한 dense prediction 태스크에서는 순수 ViT 구조가 CNN 기반 backbone보다 불리한 경우가 있어, 이를 보완하는 후속 구조들이 등장했습니다.
위치 임베딩의 한계: 학습 시 사용한 이미지 해상도와 다른 해상도로 추론할 경우 position embedding을 보간(interpolation)해야 하는 등 유연성에 제약이 있습니다.
7. CNN과의 비교 요약
항목	CNN	ViT
정보 처리 방식	지역적(local) → 계층적 확장	전역적(global), 처음부터 전체 참조
Inductive bias	강함 (지역성, 가중치 공유, 이동 불변성)	약함 (거의 없음)
데이터 효율성	적은 데이터에서도 잘 작동	대규모 데이터 필요
연산 복잡도	이미지 크기에 선형적(대략)	토큰 수 제곱에 비례 (O(n²))
확장성(스케일링)	상대적으로 완만	데이터/모델 키울수록 성능 뚜렷이 향상
대표 강점	효율성, 안정적 성능	대규모 사전학습 시 SOTA, 멀티모달 확장성
8. 이후 발전된 변형 모델들

ViT의 한계를 보완하기 위한 후속 연구들도 함께 알아두면 이해가 깊어집니다.

DeiT (Data-efficient Image Transformer): distillation 기법을 활용해 ImageNet 단독으로도 경쟁력 있는 성능을 내도록 데이터 효율성을 개선.
Swin Transformer: 이미지를 window 단위로 나누어 그 안에서만 attention을 계산하고(계산량 감소), window를 계층적으로 shift/병합하며 CNN처럼 다중 스케일 특징을 얻는 구조. detection/segmentation에서 강세.
Hybrid ViT: 초반부는 CNN으로 지역 특징을 뽑고, 이를 patch로 Transformer에 넣는 CNN+Transformer 혼합 구조.
MAE (Masked Autoencoder): 이미지 patch의 일부(75% 등)를 마스킹하고 복원하도록 self-supervised 학습, ViT의 사전학습 데이터 의존성 문제를 데이터 레이블 없이 완화.
DINO/DINOv2: self-supervised 방식으로 학습된 ViT가 별도 fine-tuning 없이도 뛰어난 특징 표현력을 보임.
CLIP: ViT를 이미지 인코더로 사용해 텍스트-이미지 쌍 대조학습(contrastive learning)을 수행, 현재 멀티모달 모델들의 기반이 되는 구조.
9. 정리

ViT의 핵심 통찰은 "이미지도 결국 시퀀스로 표현할 수 있고, 그렇다면 시퀀스 처리에 강력한 Transformer의 self-attention을 그대로 적용할 수 있다"는 것입니다. CNN이 사람이 설계한 '이미지는 지역적 구조를 가진다'는 사전 지식을 구조에 내장한 반면, ViT는 그런 가정을 최소화하고 대신 방대한 데이터로부터 스스로 학습하게 만들었습니다. 그 결과 데이터가 충분할 때는 더 뛰어난 일반화 성능과 확장성을 보이지만, 데이터가 부족하거나 연산 자원이 제한적인 환경, 혹은 세밀한 지역 정보가 중요한 태스크에서는 여전히 CNN이나 이를 절충한 Swin Transformer 같은 구조가 실용적으로 선호되기도 합니다. 현재는 순수 ViT보다는 위에서 언급한 다양한 변형·혼합 구조, 그리고 self-supervised 사전학습 기법과 결합된 형태가 실무에서 더 널리 쓰이고 있습니다.