DROP TABLE IF EXISTS temp_cohort
DROP TABLE IF EXISTS temp_cohort_all
DROP TABLE IF EXISTS temp_cohort_case

SELECT person_id AS subject_id, cohort_start_date
INTO TEMP temp_cohort_total
FROM @target_database_schema.@target_person_table_total

SELECT person_id AS subject_id, min(cohort_start_date) AS cohort_start_date
INTO TEMP temp_cohort_case
FROM @target_database_schema.@target_person_table_case
GROUP BY person_id

DROP TABLE IF EXISTS temp_cohort

SELECT t.subject_id, t.cohort_start_date, c.cohort_start_date AS cohort_start_date2
INTO TEMP temp_cohort
FROM (SELECT * FROM #temp_cohort_total) t                                  ---- (cohort number 수정)
LEFT JOIN (SELECT * FROM #temp_cohort_case) c								---- (cohort number 수정)
ON t.subject_id = c.subject_id

--SELECT * FROM #temp_cohort

---- 2) calc day diff ----

DROP TABLE IF EXISTS temp_cohort_daydiff

SELECT *, DATEDIFF(day, cohort_start_date, cohort_start_date2) AS daydiff 
INTO TEMP temp_cohort_daydiff
FROM #temp_cohort

---- 3) filtering daydiff ----

--SELECT * FROM #temp_cohort_daydiff

DROP TABLE IF EXISTS temp_cohort_daydiff_valid

SELECT * 
INTO TEMP temp_cohort_daydiff_valid
FROM #temp_cohort_daydiff WHERE daydiff = 0 or daydiff IS NULL

--SELECT * FROM #temp_cohort_daydiff_valid

---- 4) choose cohort start date ----

DROP TABLE IF EXISTS temp_cohort_dropduplicate

SELECT subject_id
, min(cohort_start_date) AS cohort_start_date
, min(cohort_start_date2) AS cohort_start_date2
, min(daydiff) AS daydiff
INTO TEMP temp_cohort_dropduplicate
FROM #temp_cohort_daydiff_valid
GROUP BY subject_id

--SELECT * --FROM #temp_cohort_dropduplicate

---- 5) join table (cohort & person) ----

DROP TABLE IF EXISTS temp_person

SELECT * 
INTO TEMP temp_person
FROM #temp_cohort_dropduplicate
LEFT JOIN @cohort_database_schema.person
ON #temp_cohort_dropduplicate.subject_id = @cohort_database_schema.person.person_id

---- 6) save table (person table) ----

DROP TABLE IF EXISTS @target_database_schema.@target_person_table

SELECT person_id, cohort_start_date, gender_source_value, (date_part('year', cohort_start_date)-year_of_birth) as age
INTO @target_database_schema.@target_person_table
FROM #temp_person

-- SELECT * FROM @target_database_schema.@target_person_table
