### Numpy 기본 사용법

## Numpy란? 
# 다차원 배열을 효과적으로 처리할 수 있도록 도와주는 도구
# python의 기본 List에 비해 빠르고 강력한 기능 제공

## Numpy의 차원
# 1차원 축(행): axis 0 => Vector
# 2차원 축(열): axis 1 => Matrix
# 3차원 축(채널): axis 2 => Tensor(3차원 이상)


## import numpy library 
import numpy as np



## python list to numpy 
list_data=[1,2,3]
array_0=np.array(list_data)


## numpy 배열 초기화
# 0부터 3까지의 배열 만들기
array_1=np.arange(4)

# 4x4 배열 값 0, dtype=float으로 초기화
array_2=np.zeros((4,4), dtype=float)

# 1로 초기화 
array_3=np.ones(5,dtype=int)

# 0부터 9까지 랜덤한 int값으로 초기화 된 3x3 배열 만들기 
array_4=np.random.randint(0,10,(3,3))

# 평균이 0이고, 표준편차가 1인 표준 정규를 띄는 배열(표준 정규 분포)
array_5=np.random.normal(0,1,(3,3))

## 배열 형태 바꾸기
array_6=np.array([1,2,3,4])
array_7=array_6.reshape((2,2))


## 배열 가로 축으로 합치기 
array_a=np.array([1,2,3])
array_b=np.array([4,5,6])
array_ab=np.concatenate([array_a,array_b])

## 배열 세로 축으로 합치기
array_a=np.arange(4).reshape(1,4) # 1행 4열
array_b=np.arange(8).reshape(2,4) # 2행 4열
array_ab=np.concatenate([array_a,array_b],axis=0)


## 배열 나누기 
array_lr=np.arange(8).reshape(2,4)
left,right=np.split(array_lr,[2],axis=1) # array_lr을 index 2 열 기준(axis=1 => 열)으로 나눈다 


### numpy 연산 

## numpy 상수 연산
array=np.random.randint(1,10,size=4).reshape(2,2)
array=array+10 # 각 배열의 값에 +10
array=array*10 # 각 배열의 값에 *10

## 서로 다른 형태의 numpy 연산
# numpy는 서로 다른 형태의 배열을 연산할 때는 행 우선으로 수행 (브로드캐스트: 형태가 다른 배열을 연산할 수 있도록 배열의 형태를 동적으로 변환)
array_a=np.arange(4).reshape(2,2) # (2x2)
array_b=np.arange(2) # (1x2)

array_ab=array_a+array_b


## numpy의 마스킹 연산
# 마스킹: 각 원소에 대하여 체크 후 True, False 값 
# 조건문으로 반복 조회하는 것보다 효율적 
array=np.arange(16).reshape(4,4)
array_masking=array<10 # array의 모든 원소에 대해 10보다 작은지 체크 후 T,F 
array[array_masking]=100 # array에서 10보다 작은 원소들만 100으로 값 변경 

## numpy의 집계 함수
# 최대값(np.max(array)), 최소값(np.min(array)), 합계(np.sum(array)), 평균(np.mean(array))
array=np.arange(16).reshape(4,4)
max_value=np.max(array)

# 특정 축 기준으로 집계 함수 수행(ex: 특정 열에 대해서면 집계)
array=np.arange(16).reshape(4,4)
max_value=np.max(array,axis=0) # 각 열에 대해서 모든 행의 값을 비교하여 가장 큰 수 집계


### numpy의 활용

## numpy의 저장과 불러오기 
## numpy 배열을 파일 형태로 저장하거나 다시 가져올 수 있다(확장자: .npy)

# 단일 객체 저장 및 불러오기 
# array=np.arange(0,10)
# np.save('saved.npy',array) # 저장

# result=np.load('saved.npy') # 불러오기 

# 복수 객체 저장 및 불러오기 
# array_1-np.arange(0,10)
# array_2=np.arange(10,20)
# np.savez('saved.npz',array_a=array_1,array_b=array_2) # array_a에 array_1, array_b에 array_2 저장 

# data=np.load('saved.npz') # 복수 객체 불러오기 
# result_1=data['array_a'] # 저장했던 이름 인덱스로 접근 
# result_2=data['array_b']

## numpy 원소의 정렬 
# 기본 설정 = 오름차순 정렬
# numpy 원소 오름차순 정렬 
array=np.array([5,9,10,3,1])
array.sort() # 오름차순 정렬 

# print(array[::-1]) # 내림차순 정렬 

# 각 열을 기준으로 정렬 
array=np.array([5,9,10,3,1],[8,3,4,2,5])
array.sort(axis=0) # 각각 열 기준으로 오름차순 정렬 

## 균일한 간격으로 데이터 생성
# linspace: 시작 값과 끝 값 사이에 몇 개의 데이터가 있는지
array=np.linspace(0,10,5) # 0~10 사이를 5개의 데이터가 채우도록 데이터 생성 [0. 2.5 5. 7.5 10.  ]

## 난수의 재연 (실행마다 결과 동일)
# 난수는 실행 시 마다 값이 변경 -> seed 설정으로 항상 동일한 난수 값으로 설정 
np.random.seed(7) # seed 설정으로 난수 값이 변경되어서 학습 결과가 달라지는 일 없도록 한다 

## numpy 배열 객체 복사 
array_1=np.arange(0,10)
# array_2=array_1 # 같은 주소값 지정 
# array_2[2]=99 # array_2 값을 변경해도 array_1값도 변경 -> 포인터로 연결되어 있다 

array_2=array_1.copy() # array_1의 값만 복사, 같은 주소값 x -> array_2 값 변경해도 array_1 값 변경되지 않는다

## 중복된 원소 제거 
array=np.array([1,1,2,2,2,3,3,4])
np.unique(array) # 중복된 원소 제거 