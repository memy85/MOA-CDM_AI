-- 1. drug concepts
IF OBJECT_ID('tempdb..#temp_drug_concept') IS NOT NULL
	DROP TABLE #temp_drug_concept

SELECT DISTINCT concept_id, concept_name
INTO #temp_drug_concept
FROM @cohort_database_schema.drug_exposure d
LEFT JOIN @cohort_database_schema.concept c
ON d.drug_concept_id = c.concept_id

-- 2, search string
IF OBJECT_ID('tempdb..#temp_concepts') IS NOT NULL
	DROP TABLE #temp_concepts

SELECT *
INTO #temp_concepts
FROM #temp_drug_concept
WHERE concept_name LIKE '%@drugname%'

-- 3, results 
-- SELECT * FROM #temp_concepts

-- 4, make a list
SELECT DISTINCT STUFF((SELECT ',' + CAST(concept_id AS varchar) 
	FROM #temp_concepts 
	FOR XML PATH('') ), 1, 1, '') AS concept_list
FROM #temp_concepts 