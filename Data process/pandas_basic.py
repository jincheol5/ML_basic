### Pandas란?
# 데이터를 효과적으로 처리하고, 보여줄 수 있도록 도와주는 라이브러리
# Numpy와 함께 사용되어 다양한 연계적인 기능 제공
# 인덱스(index)에 따라 데이터를 나열하므로 사전(Dictionary) 자료형에 가깝다
# 시리즈(series)를 기본적인 자료형으로 사용 (frame=db의 table 전체, series=db의 특정 속성 열 전체이며 행에 대한 index 가짐)
# 엑셀(Excel)과 거의 똑같다 

## Series란?
# 인덱스와 값으로 구성 


import pandas as pd
import numpy as np


## pandas 기본 사용법
# series 생성
array=pd.Series(['apple','banana','carrot'],index=['a','b','c'])
# print(array)
# print(array['a']) # 특정 index에 해당하는 값 출력 

# dictionary to series
data={
    'a':'apple',
    'b':'banana',
    'c':'carrot'
}
array=pd.Series(data)

## Data frame이란?
# 다수의 시리즈(series)를 모아 처리하기 위한 목적으로 사용
# 표 형태로 데이터를 손쉽게 출력하고 할 때 사용할 수 있다

# data frame 생성
word_dict={
    'a':'apple',
    'b':'banana',
    'c':'carrot'
}

frequency_dict={
    'a':3,
    'b':5,
    'c':7
}

word=pd.Series(word_dict)
frequency=pd.Series(frequency_dict)

# 이름(name):값(value)
summary=pd.DataFrame({
    'word':word,
    'frequency':frequency
})

# print(summary)

## series 연산 
# series 곱셈 
word_dict={
    'a':'apple',
    'b':'banana',
    'c':'carrot'
}

frequency_dict={
    'a':3,
    'b':5,
    'c':7
}

importance_dict={
    'a':3,
    'b':2,
    'c':1
}

word=pd.Series(word_dict)
frequency=pd.Series(frequency_dict)
importance=pd.Series(importance_dict)

summary=pd.DataFrame({
    'word':word,
    'frequency':frequency,
    'importance':importance
})

score=summary['frequency']*summary['importance'] #각 series에 같은 index 끼리 연산
summary['score']=score
# print(summary)

## data frame 슬라이싱
word_dict={
    'a':'apple',
    'b':'banana',
    'c':'carrot',
    'd':'durian'
}

frequency_dict={
    'a':3,
    'b':5,
    'c':7,
    'd':2
}

importance_dict={
    'a':3,
    'b':2,
    'c':1,
    'd':1
}

word=pd.Series(word_dict)
frequency=pd.Series(frequency_dict)
importance=pd.Series(importance_dict)

summary=pd.DataFrame({
    'word':word,
    'frequency':frequency,
    'importance':importance
})

# 이름을 기준으로 슬라이싱
# print(summary.loc['b':'c','importance':]) # series index b 부터 c 까지, dataframe index importance부터 모두 

# # 인덱스를 기준으로 슬라이싱
# print(summary.iloc[1:3,2:]) # series index 1부터 2(3-1)까지, dataframe index 2부터 모두 

## data frame의 연산

# 데이터의 변경
summary.loc['a','importance']=5 # series index 'a', dataframe index 'importance'값을 5로 변경 

# 새 데이터 삽입
summary.loc['e']=['elderberry',5,3] # 모든 series 개수 맞춰줘야 함 

## Excel로 내보내기/불러오기
# summary.to_csv("summary.csv",encoding="utf-8-sig") # excel 파일로 저장 -> DataFrame class의 메서드
# saved=pd.read_csv("summary.csv",index_col=0) # excel 파일 읽어오기 -> pandas class의 메서드 

### Pandas의 연산과 함수

## dataframe의 NULL여부 확인
# NULL값은 np.nan이나 NULL 사용

word_dict={
    'a':'apple',
    'b':'banana',
    'c':'carrot',
    'd':'durian'
}

frequency_dict={
    'a':3,
    'b':5,
    'c':np.nan, # NULL, not a number
    'd':2
}

importance_dict={
    'a':3,
    'b':2,
    'c':1,
    'd':1
}

word=pd.Series(word_dict)
frequency=pd.Series(frequency_dict)
importance=pd.Series(importance_dict)

summary=pd.DataFrame({
    'word':word,
    'frequency':frequency,
    'importance':importance
})

# print(summary.notnull()) # null 값이 아닌 데이터만 true로, 나머지 false
# print(summary.isnull()) # null 값인 데이터만 true로, 나머지 false
# summary['frequency']=summary['frequency'].fillna('no data') # nan값인 데이터를 'no data' 값으로 변경
# print(summary) 

# series 자료형의 연산 with null
array_1=pd.Series([1,2,3],index=['A','B','C'])
array_2=pd.Series([4,5,6],index=['B','C','D'])

array=array_1.add(array_2,fill_value=0) # series index가 겹치지 않는 부분은(데이터가 없는 공간) 0으로 계산해서 연산
# print(array)

## dataframe 자료형의 연산 
# dataframe 생성 시 dataframe index 지정 안하면 0,1,2,...로 자동 지정 
array_1=pd.DataFrame([[1,2],[3,4]],index=['A','B']) # 여기서 index는 series index
array_2=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],index=['B','C','D'])

array=array_1.add(array_2,fill_value=0) # 겹치지 않는 부분은 0으로 연산, 둘 다 없는 공간은 NaN 처리  
# print(array)

## dataframe 집계 함수
# print(array[1].sum()) # column 1의 합

## dataframe 정렬 함수
word_dict={
    'a':'apple',
    'b':'banana',
    'c':'carrot',
    'd':'durian'
}

frequency_dict={
    'a':3,
    'b':5,
    'c':7,
    'd':2
}

importance_dict={
    'a':3,
    'b':2,
    'c':1,
    'd':1
}

word=pd.Series(word_dict)
frequency=pd.Series(frequency_dict)
importance=pd.Series(importance_dict)

summary=pd.DataFrame({
    'word':word,
    'frequency':frequency,
    'importance':importance
})

# 정렬
summary=summary.sort_values('frequency',ascending=False) # datafame index frequency 기준으로 내림차순 정렬
# print(summary)

### Pandas의 활용

## dataframe의 마스킹 
df=pd.DataFrame(np.random.randint(1,10,(2,2)),index=[0,1],columns=['A','B'])
# print(df)

# 마스킹 연산
# print(df['A']<=5) # column A의 각 원소가 5보다 작거나 같은지 true false로 출력 
# print(df.query("A <= 5 and B <= 8")) # column A의 원소가 5보다 작고, column B의 원소가 8보다 작은 행 추출

## dataframe의 개별 연산 
df=pd.DataFrame([[1,2,3,4],[1,2,3,4]],index=[0,1],columns=['A','B','C','D'])

# 함수 적용
df=df.apply(lambda x:x+1) # 함수에 대한 정보 넣었을 경우 함수에 따라 모든 데이터에 적용 

def addOne(x):
    return x+1

df=df.apply(addOne)

# 특정 index의 데이터 값 변경
db=pd.DataFrame([
    ['apple','apple','carrot','banana'],['durian','banana','apple','carrot']],
    index=[0,1],
    columns=['A','B','C','D'])

df=df.replace({'apple':'airport'})

## dataframe의 그룹화 

# groupby
df=pd.DataFrame([
    ['apple',7,'fruit'],['banana',3,'fruit'],['beef',5,'meal'],['kimchi',4,'meal']],
    columns=['name','frequency','type'])

df.groupby(['type']).sum() # type에 대해 그룹화 후 sum 가능한 정수형 데이터들만 sum  

# aggregate
df=pd.DataFrame([
    ['apple',7,5,'fruit'],
    ['banana',3,6,'fruit'],
    ['beef',5,2,'meal'],
    ['kimchi',4,8,'meal']],
    columns=['name','frequency','importance','type'])

# print(df)
# print(df.groupby(['type']).aggregate([min,max,np.average])) # 여러개의 groupby 연산 한번에 수행

# 필터링
def my_filter(data):
    return data['frequency'].mean() >= 5 # 그룹화 후 평균값 5 이상인 것만 출력 

# df=df.groupby('type').filter(my_filter)
# print(df)

df['gap']=df.groupby('type')['frequency'].apply(lambda x:x-x.mean()) # type 그룹별로 frequency값이 frequecy의 mean값과 얼마나 차이나는지 gap에 저장

# print(df)

## dataframe의 다중화 
# index를 다중화하여 설정 
df=pd.DataFrame(
    np.random.randint(1,10,(4,4)),
    index=[['1_try','1_try','2_try','2_try'],['attack','defense','attack','defense']], # index에 대해 2차원 리스트 형태로 입력
    columns=['one','two','three','four']
)

# print(df) 

# 다중화에 대한 연산 처리 
# print(df[['one','two']].loc['2_try']) # one, two 컬럼에 대해서 2_try에 대해서만 출력 

## 피벗 테이블의 기초 
# 열과 행을 서로 바꾼다던지 하는 다양한 처리 
df=pd.DataFrame([
    ['apple',7,5,'fruit'],
    ['banana',3,6,'fruit'],
    ['coconut',2,6,'fruit'],
    ['rice',8,2,'meal'],
    ['beef',5,2,'meal'],
    ['kimchi',4,8,'meal']],
    columns=['name','frequency','importance','type'])

print(df)

df=df.pivot_table(
    index='importance',columns='type',values='frequency',
    aggfunc=np.max    
)
print(df)