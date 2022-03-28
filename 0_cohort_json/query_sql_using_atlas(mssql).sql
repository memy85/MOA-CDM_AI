-- 0. �����ؾ��� �� : 
--		1) table name ( cohort : ex.WEBAPI.dbo.cohort / person : ex. CDM.dbo.person / result : ex. temp.dbo.person_meloxicam )
--			: meloxicam, celecoxib, valproic_acid, lamotrigine
--		2) cohort number (Atlas�� ����� Cohort ID (*Target�ϰ� Abnormal �ٲ��� �ʵ��� ����))
--			: meloxicam(Target) 		:            ex) 1405
--			: meloxicam(abnormal) 		:            ex) 1406
--			: celecoxib(Target) 		:            ex) 1407
--			: celecoxib(abnormal) 		:            ex) 1408
--			: valproic_acid(Target)		:            ex) 1409
--			: valproic_acid(abnormal)	:            ex) 1410
--			: lamotrigine(Target) 		:            ex) 1411
--			: lamotrigine(abnormal) 	:            ex) 1412
-- 1. cohort ����
--		t ; target (case + control)	; cohort_start_date = index_date
--		c ; case					; cohort_start_date = first_abnormal_date

if OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	drop table #temp_cohort

select t.subject_id, t.cohort_start_date, c.cohort_start_date as first_abnormal_date
into #temp_cohort
from (select * from WEBAPI.dbo.cohort where cohort_definition_id = 1406) t                                  ---- (cohort number ����)
left join (select * from WEBAPI.dbo.cohort where cohort_definition_id = 1405) c								---- (cohort number ����)
on t.subject_id = c.subject_id

-- 1-1. ��� Ȯ�� 

-- 2. cohort�� person ����
if OBJECT_ID('tempdb..#temp_person') IS NOT NULL
	drop table #temp_person

select * 
into #temp_person
from #temp_cohort
left join CDM.dbo.person
on #temp_cohort.subject_id = CDM.dbo.person.person_id

-- 2-1. subject_id, cohort_start_date, first_abnormal_date, age, gender_source_value
if OBJECT_ID('temp.dbo.person_meloxicam') IS NOT NULL															---- (table name ����)
	drop table temp.dbo.person_meloxicam																		---- (table name ����)

--select person_id, cohort_start_date, gender_source_value,
--(YEAR(cohort_start_date)-year_of_birth) as age,
--first_abnormal_date
--into temp.dbo.person_meloxicam
--from #temp_person		

select person_id, cohort_start_date, gender_source_value,
(YEAR(cohort_start_date)-year_of_birth) as age,
(case when DATEDIFF(day, cohort_start_date, first_abnormal_date) > 60	then NULL
	  when DATEDIFF(day, cohort_start_date, first_abnormal_date) < 1	then NULL
																		else first_abnormal_date 
end) as first_abnormal_date	
into temp.dbo.person_meloxicam																					---- (table name ����)
from #temp_person

select * from temp.dbo.person_meloxicam where first_abnormal_date is not null
		
-- �ӽ� ���̺� ����
if OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	drop table #temp_cohort
if OBJECT_ID('tempdb..#temp_person') IS NOT NULL
	drop table #temp_person

--select case when DATEDIFF(day, first_abnormal_date, cohort_start_date) > 60 then first_abnormal_date else NULL end from #temp_person

--DATEDIFF(Day, '2017/08/25', '2017/10/25')
