# Tokenize_Wordcount

### Data
  
|데이터|긍정(1)|부정(0)|
|:------:|:---:|:---:|
|리뷰 데이터|82598|10218|



### Requirements
- pandas
- imblearn
- re
- konlpy
- sklearn
- nltk
- matplotlib
- numpy
- scikitplot


### directory & file structure.
```
.
|
├── Konlpy_Hannanum.ipynb 
├── Konlpy_Kkma.ipynb
├── Konlpy_Mecab.ipynb
├── Konlpy_Okt.ipynb
├── Konlpy_Twitter.ipynb
└── README.md
```

     
### Data Preparation
- data imbalanced를 해결하기 위해 Random Under Sample을 사용   
- emoji와 특수문자 제거       
- 문장 단위의 토큰을 하나의 리스트로 모아서 모든 리뷰의 문장의 토큰을 확인   
- 데이터에서 자주 사용되는 토큰 10000개를 세어 인코딩한 벡터를 생성


### KoNLPy Tokenizer

##### Hannanum
- 한나눔. KAIST Semantic Web Research Center 개발
- http://semanticweb.kaist.ac.kr/hannanum/

|Tag|Description|
|:---:|:--------:|
|N|체언|
|P|용언|
|M|수식언|
|I|독립언|
|J|관계언|
|E|어미|
|X|접사|
|S|기호|
|F|외국어|

     
##### Kkma
- 꼬꼬마. 서울대학교 IDS(Intelligent Data Systems) 연구실 개발
- http://kkma.snu.ac.kr/

|Tag|Description|
|:---:|:--------:|
|NNG|보통명사|
|NNP|고유명사|
|NNB|일반 의존 명사|
|NNM|단위 의존 명사|
|NR|수사|
|NP|대명사|
|VV|동사|
|VA|형용사|
|VXV|보조 동사|
|VXA|보조 형용사|
|VCP|긍정 지정사, 서술격 조사 '이다'|
|VCN|부정 지정사, 형용사 '아니다'|
|MDN|수 관형사|
|MDT|일반 관형사|
|MAG|일반 부사|
|MAC|접속 부사|
|IC|감탄사|
|JKS|주격 조사|
|JKC|보격 조사|
|JKG|관형격 조사|
|JKO|목적격 조사|
|JKM|부사격 조사|
|JKI|호격 조사|
|JKQ|인용격 조사|
|JC|접속 조사|
|JX|보조사|
|EPH|존칭 선어말 어미|
|EPT|시제 선어말 어미|
|EPP|공손 선어말 어미|
|EFN|평서형 종결 어미|
|EFQ|의문형 종결 어미|
|EFO|명령형 종결 어미|
|EFA|청유형 종결 어미|
|EFI|감탄형 종결 어미|
|EFR|존칭형 종결 어미
|ECE|대등 연결 어미|
|ECS|보조적 연결 어미|
|ECD|의존적 연결 어미|
|ETN|명사형 전성 어미|
|ETD|관형형 전성 어미|
|XPN|체언 접두사|
|XPV|용언 접두사|
|XSN|명사파생 접미사|
|XSV|동사 파생 접미사|
|XSA|형용사 파생 접미사|
|XR|어근|
|SF|마침표, 물음표, 느낌표|
|SE|줄임표|
|SS|따옴표,괄호표,줄표|
|SP|쉼표,가운뎃점,콜론,빗금|
|SO|붙임표(물결,숨김,빠짐)|
|SW|기타기호 (논리수학기호,화폐기호)|
|OH|한자|
|OL|외국어|
|ON|숫자|
|UN|명사추정범주|
     
##### Mecab
- 메카브. 일본어용 형태소 분석기를 한국어를 사용할 수 있도록 수정
- https://bitbucket.org/eunjeon/mecab-ko

|Tag|Description|
|:---:|:--------:|
|NNG|일반 명사|
|NNP|고유 명사|
|NNB|의존 명사|
|NNBC|단위를 나타내는 명사|
|NR|수사|
|NP|대명사|
|VV|동사|
|VA|형용사|
|VX|보조 용언|
|VCP|긍정 지정사|
|VCN|부정 지정사|
|MM|관형사|
|MAG|일반 부사|
|MAJ|접속 부사|
|IC|감탄사|
|JKS|주격 조사|
|JKC|보격 조사|
|JKG|관형격 조사|
|JKO|목적격 조사|
|JKB|부사격 조사|
|JKV|호격 조사|
|JKQ|인용격 조사|
|JC|접속 조사|
|JX|보조사|
|EP|선어말어미|
|EF|종결 어미|
|EC|연결 어미|
|ETN|명사형 전성 어미|
|ETM|관형형 전성 어미|
|XPN|체언 접두사|
|XSN|명사파생 접미사|
|XSV|동사 파생 접미사|
|XSA|형용사 파생 접미사|
|XR|어근|
|SF|마침표, 물음표, 느낌표|
|SE|줄임표 …|
|SSO|여는 괄호 (, [|
|SSC|닫는 괄호 ), ]|
|SC|구분자 , · / :|
|SY|기타 기호|
|SH|한자|
|SL|외국어|
|SN|숫자|
     
##### Okt
- 오픈 소스 한국어 분석기
- https://github.com/open-korean-text/open-korean-text

|Tag|Description|
|:---:|:--------:|
|Noun|명사|
|Verb|동사|
|Adjective|형용사|
|Determiner|관형사|
|Adverb|부사|
|Conjunction|접속사|
|Exclamation|감탄사|
|Josa|조사|
|PreEomi|선어말어미|
|Eomi|어미|
|Suffix|접미사|
|Punctuation|구두점|
|Exclamation|감탄사|
|Foreign|외국어, 한자 및 기타기호|
|Alpha|알파벳|
|Number|숫자|
|Unknown|미등록어|
|KoreanParticle|(ex: ㅋㅋ)|
|Hashtag|트위터 해쉬태그|
|ScreenName|트위터 아이디|
      
##### Twitter
- 트위터 형태소 분석기
- https://github.com/twitter/twitter-korean-text

|Tag|Description|
|:---:|:--------:|
|Noun|명사|
|Verb|동사|
|Adjective|형용사|
|Determiner|관형사|
|Adverb|부사|
|Conjunction|접속사|
|Exclamation|감탄사|
|Josa|조사|
|PreEomi|선어말어미|
|Eomi|어미|
|Suffix|접미사|
|Punctuation|구두점|
|Foreign|외국어, 한자 및 기타기호|
|Alpha|알파벳|
|Number|숫자|
|Unknown|미등록어|
|KoreanParticle|(ex: ㅋㅋ)|
|Hashtag|트위터 해쉬태그|
|ScreenName|트위터 아이디|
|Email|이메일 주소|
|URL|웹주소|
     
### Modeling
- Naive Bayes 적용

### Result

|Tokenizer|token 개수|모델 정확도|recall|f1-score|
|:--------:|:-------:|:-------:|:-----:|:------:|
|Hannanum|1080349|0.7282|0.7791|0.7412|
|Kkma|1312773|0.7309|0.7502|0.7388|
|Mecab|1179471|0.7441|0.7770|0.7507
|Okt|915367|0.7461|0.7873|0.7604|
|Twitter|908537|0.7341|0.7753|0.7457|
