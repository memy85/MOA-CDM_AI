### CDM 기반 의약품 부작용 예측 모델 생성
### 간독성 / 신독성 예측 모델 

순서 

0_cohort_json
   - Atlas에서 Cohort 생성 
   - Target / Abnoraml Cohort 생성
   - DB에 Table 생성
   : .\0_cohort_json\README.md 참고

1_importsql 
   - DB에서 데이터를 불러와 파일로 저장 
   - .\1_importsql\0_readDB.ipynb 실행

2_preprocessing
   - 생성된 Table을 읽어와 유효변수 추출 및 Normalization 등 진행 
   - .\2_preprocessing\0_preprocessing.ipynb 실행

3_xgboost_classification
   - 전처리된 데이터를 읽어와 XGBoost로 예측 모델 생성 
   - .\3_xgboost_classification\0_xgboost.ipynb 실행
   - .\3_xgboost_classification\0_imxgboost.ipynb 실행

4_code_data_visualization
   - CDM 데이터 품질 확인 
   - Lab Test 별 Age, sex별 Lab value 분포 확인 
   - .\4_code_data_visualization\cdm_visualization_query.sql
   - .\4_code_data_visualization\0_data_visualization.ipynb 실행

결과 확인
   - data (모델 생성을 위한 데이터 저장)
   - result (결과 추출을 위한 데이터 저장)

: result 폴더만 압축해서 전달 요청
