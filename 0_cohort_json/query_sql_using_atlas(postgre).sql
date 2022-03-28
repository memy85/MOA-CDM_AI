-- 0. 변경해야할 것 : 
--		1) table name ( cohort : ex.WEBAPI.dbo.cohort / person : ex. CDM.dbo.person / result : ex. temp.dbo.person_meloxicam )
--			: meloxicam, celecoxib, valproic_acid, lamotrigine
--		2) cohort number (Atlas에 등록한 Cohort ID (*Target하고 Abnormal 바뀌지 않도록 주의))
--			: meloxicam(Target) 		:            ex) 1405
--			: meloxicam(abnormal) 		:            ex) 1406
--			: celecoxib(Target) 		:            ex) 1407
--			: celecoxib(abnormal) 		:            ex) 1408
--			: valproic_acid(Target)		:            ex) 1409
--			: valproic_acid(abnormal)	:            ex) 1410
--			: lamotrigine(Target) 		:            ex) 1411
--			: lamotrigine(abnormal) 	:            ex) 1412
-- 1. cohort 병합
--		t ; target (case + control)	; cohort_start_date = index_date
--		c ; case					; cohort_start_date = first_abnormal_date

DROP TABLE IF EXISTS temp_cohort;

select t.subject_id, t.cohort_start_date, c.cohort_start_date as first_abnormal_date
into temp temp_cohort
from (select * from WEBAPI.dbo.cohort where cohort_definition_id = 1405) t                                  ---- (cohort number 수정)
left join (select * from WEBAPI.dbo.cohort where cohort_definition_id = 1406) c								---- (cohort number 수정)
on t.subject_id = c.subject_id

-- 1-1. 결과 확인 

-- 2. cohort와 person 병합
DROP TABLE IF EXISTS temp_person;

select * 
into temp temp_person
from temp_cohort
left join CDM.dbo.person
on temp_cohort.subject_id = CDM.dbo.person.person_id

-- 2-1. subject_id, cohort_start_date, first_abnormal_date, age, gender_source_value
if OBJECT_ID('temp.dbo.person_meloxicam') IS NOT NULL															---- (table name 수정)
	drop table temp.dbo.person_meloxicam																		---- (table name 수정)

select person_id, cohort_start_date, first_abnormal_date, (date_part('year', cohort_start_date)-year_of_birth) as age, gender_source_value
into temp.dbo.person_meloxicam																				---- (table name 수정)
from temp_person

-- 임시 테이블 제거
DROP TABLE IF EXISTS temp_cohort;

DROP TABLE IF EXISTS temp_person;