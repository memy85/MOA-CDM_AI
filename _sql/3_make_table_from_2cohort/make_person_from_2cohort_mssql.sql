IF OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	DROP TABLE #temp_cohort

IF OBJECT_ID('tempdb..#temp_cohort_all') IS NOT NULL
	DROP TABLE #temp_cohort_all

IF OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	DROP TABLE #temp_cohort_case

SELECT person_id AS subject_id, cohort_start_date
INTO #temp_cohort_total
FROM @target_database_schema.@target_person_table_total

SELECT person_id AS subject_id, min(cohort_start_date) AS cohort_start_date
INTO #temp_cohort_case
FROM @target_database_schema.@target_person_table_case
GROUP BY person_id

IF OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	DROP TABLE #temp_cohort

SELECT t.subject_id, t.cohort_start_date, c.cohort_start_date AS cohort_start_date2
INTO #temp_cohort
FROM (SELECT * FROM #temp_cohort_total) t
LEFT JOIN (SELECT * FROM #temp_cohort_case) c
ON t.subject_id = c.subject_id

--SELECT * FROM #temp_cohort

---- 2) calc day diff ----

IF OBJECT_ID('tempdb..#temp_cohort_daydiff') IS NOT NULL
	DROP TABLE #temp_cohort_daydiff

SELECT *, DATEDIFF(day, cohort_start_date, cohort_start_date2) AS daydiff 
INTO #temp_cohort_daydiff
FROM #temp_cohort

---- 3) filtering daydiff ----

--SELECT * FROM #temp_cohort_daydiff

IF OBJECT_ID('tempdb..#temp_cohort_daydiff_valid') IS NOT NULL
	DROP TABLE #temp_cohort_daydiff_valid

SELECT * 
INTO #temp_cohort_daydiff_valid
FROM #temp_cohort_daydiff where daydiff = 0 or daydiff is NULL

--SELECT * FROM #temp_cohort_daydiff_valid

---- 4) choose cohort start date ----

IF OBJECT_ID('tempdb..#temp_cohort_dropduplicate') IS NOT NULL
	DROP TABLE #temp_cohort_dropduplicate

SELECT subject_id
, min(cohort_start_date) AS cohort_start_date
, min(cohort_start_date2) AS cohort_start_date2
, min(daydiff) AS daydiff
INTO #temp_cohort_dropduplicate
FROM #temp_cohort_daydiff_valid
GROUP BY subject_id

--SELECT * --FROM #temp_cohort_dropduplicate

---- 5) join table (cohort & person) ----

IF OBJECT_ID('tempdb..#temp_person') IS NOT NULL
	DROP TABLE #temp_person

SELECT * 
INTO #temp_person
FROM #temp_cohort_dropduplicate
LEFT JOIN @cohort_database_schema.person
ON #temp_cohort_dropduplicate.subject_id = @cohort_database_schema.person.person_id

---- 6) save table (person table) ----

IF OBJECT_ID('@target_database_schema.@target_person_table') IS NOT NULL
	DROP TABLE @target_database_schema.@target_person_table

SELECT person_id, cohort_start_date, (YEAR(cohort_start_date)-year_of_birth) AS age, gender_source_value
INTO @target_database_schema.@target_person_table
FROM #temp_person

-- SELECT * FROM @target_database_schema.@target_person_table
