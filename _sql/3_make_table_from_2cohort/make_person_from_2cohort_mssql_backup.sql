---- 1) 2 cohort ----

if OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	drop table #temp_cohort

select t.subject_id, t.cohort_start_date, c.cohort_start_date as first_abnormal_date
into #temp_cohort
from (select * from WEBAPI.dbo.cohort where cohort_definition_id = 1424) t                                  ---- (cohort number 수정)
left join (select * from WEBAPI.dbo.cohort where cohort_definition_id = 1425) c								---- (cohort number 수정)
on t.subject_id = c.subject_id

--select * from #temp_cohort

---- 2) calc day diff ----

if OBJECT_ID('tempdb..#temp_cohort_daydiff') IS NOT NULL
	drop table #temp_cohort_daydiff

select *, DATEDIFF(day, cohort_start_date, first_abnormal_date) as daydiff 
into #temp_cohort_daydiff
from #temp_cohort


---- 3) filtering daydiff ----

--select * from #temp_cohort_daydiff

if OBJECT_ID('tempdb..#temp_cohort_daydiff_valid') IS NOT NULL
	drop table #temp_cohort_daydiff_valid

select * 
into #temp_cohort_daydiff_valid
from #temp_cohort_daydiff where daydiff < 60 and daydiff > 0 or daydiff is NULL

--select * from #temp_cohort_daydiff_valid

--select subject_id, count(*)
--from #temp_cohort_daydiff_valid
--group by subject_id

---- 4) choose cohort start date ----

if OBJECT_ID('tempdb..#temp_cohort_dropduplicate') IS NOT NULL
	drop table #temp_cohort_dropduplicate

select subject_id
, min(cohort_start_date) as cohort_start_date
, min(first_abnormal_date) as first_abnormal_date
, min(daydiff) as daydiff
into #temp_cohort_dropduplicate
from #temp_cohort_daydiff_valid
group by subject_id

--select * --from #temp_cohort_dropduplicate

---- 5) join table (cohort & person) ----

if OBJECT_ID('tempdb..#temp_person') IS NOT NULL
	drop table #temp_person

select * 
into #temp_person
from #temp_cohort_dropduplicate
left join CDM.dbo.person
on #temp_cohort_dropduplicate.subject_id = CDM.dbo.person.person_id

---- 6) save table (person table) ----

select * 
into temp.dbo.acetoaminophen
from #temp_person 
