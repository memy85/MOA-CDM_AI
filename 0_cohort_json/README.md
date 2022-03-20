1. Atlas에 Cohort 생성
	1)  Atlas 열기
	2)  코호트 정의 
	3)  새 코호트 
	4)  Cohort Name 수정(Json파일명과 동일) 
	5)  내보내기 Tab 
	6)  json 
	7)  json 파일 복사 및 붙여넣기 
	8)  새로고침 
	9)  저장 (초록색 파일 아이콘)
	10) 생성 Tab 
	11) 생성 버튼 클릭 (환자 수 확인)

2. 위 Step을 파일 갯수 별로 반복 (총 8회)


3. SQL Manager(mssql or postgres) 실행

	1) 0_cohort_json/query_sql_using_atlas.sql 열기
	2) 위에서 Atlas에서 생성한 Cohort 번호 수정 
	3) Cohort Number와 Table 번호를 바꿔가면서 순차적으로 실행 
