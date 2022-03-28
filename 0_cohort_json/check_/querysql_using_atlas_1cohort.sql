-- 1. cohort 
if OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	drop table #temp_cohort

select subject_id, cohort_start_date into #temp_cohort from WEBAPI.dbo.cohort where cohort_definition_id = 1406

-- 2. cohort와 person 병합
if OBJECT_ID('tempdb..#temp_cohortperson') IS NOT NULL
	drop table #temp_cohortperson

select * 
into #temp_cohortperson
from #temp_cohort
left join CDM.dbo.person
on #temp_cohort.subject_id = CDM.dbo.person.person_id

-- 2-1. subject_id, cohort_start_date, age, gender_source_value
if OBJECT_ID('temp.dbo.person_acetaminophen') IS NOT NULL															---- (table name 수정)
	drop table temp.dbo.person_acetaminophen																		---- (table name 수정)

select person_id, cohort_start_date, gender_source_value, (YEAR(cohort_start_date)-year_of_birth) as age
into temp.dbo.person_acetaminophen																				---- (table name 수정)
from #temp_cohortperson

-- 임시 테이블 제거
if OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	drop table #temp_cohort
if OBJECT_ID('tempdb..#temp_cohortperson') IS NOT NULL
	drop table #temp_cohortperson