--# 3013721 # temp.dbo.person_AST	; AST
--# 3006923 # temp.dbo.person_ALT	; ALT
--# 3035995 # temp.dbo.person_ALP	; ALP
--# 3022217 # temp.dbo.person_PT	; PT[INR] (Prothrombin Time)
--# 3024128 # temp.dbo.person_TB	; Total bilirubin
--# 3016723 # temp.dbo.person_CR	; Creatinine

-- 0. �����ؾ��� �� : 
--		1) DB schema name 
--			: CDM.dbo.measurement
--		2) concept_id, db_name�� ¦���缭 ����
--			: ��ü 6�� Table ����

select * from cdm.dbo.measurement 
where measurement_concept_id = 3013721																		-- (concept_id ����)

-- 0. Check if use only one unit
select unit_concept_id, unit_source_value, COUNT(unit_source_value) from cdm.dbo.measurement 
where measurement_concept_id = 3013721																		-- (concept_id ����)
group by unit_concept_id, unit_source_value

-- 1. 
IF OBJECT_ID('tempdb..#temp_meas') IS NOT NULL
	DROP TABLE #temp_meas

select distinct person_id, measurement_concept_id, measurement_date, value_as_number, unit_concept_id
into #temp_meas
from cdm.dbo.measurement 
where measurement_concept_id = 3013721																		-- (concept_id ����)

---- 2-1. max���� �ִ� row ���� method (1)

--IF OBJECT_ID('tempdb..#temp_meas_group') IS NOT NULL
--	DROP TABLE #temp_meas_group

--SELECT a.*
--into #temp_meas_group
--FROM #temp_meas a
--INNER JOIN (
--    SELECT person_id, MAX(value_as_number) as value_as_number
--    FROM #temp_meas
--    GROUP BY person_id
--) b ON a.person_id = b.person_id AND a.value_as_number = b.value_as_number

-- 2-2. max���� �ִ� row ���� method (2)
IF OBJECT_ID('tempdb..#temp_meas_group') IS NOT NULL
	DROP TABLE #temp_meas_group

SELECT a.*
into #temp_meas_group
FROM #temp_meas a
LEFT JOIN #temp_meas b
ON a.person_id = b.person_id AND a.value_as_number < b.value_as_number
WHERE b.person_id IS NULL;

-- 3. 
IF OBJECT_ID('tempdb..#temp_3013721') IS NOT NULL															-- (concept_id ����)
	DROP TABLE #temp_3013721																				-- (concept_id ����)

select m.person_id, m.measurement_concept_id, m.value_as_number, p.gender_source_value, (YEAR(m.measurement_date)-p.year_of_birth) as age, p.year_of_birth, m.measurement_date
into #temp_3013721
from (
	(select * from #temp_meas_group) as m
	inner join
	(select person_id, year_of_birth, gender_source_value from CDM.dbo.person) as p
	on m.person_id = p.person_id
)

IF OBJECT_ID('temp.dbo.person_AST') IS NOT NULL																-- (DB Name ����)
	DROP TABLE temp.dbo.person_AST																			-- (DB Name ����)

select person_id, measurement_concept_id, value_as_number, gender_source_value, age
into temp.dbo.person_AST																					-- (DB Name ����)
from #temp_3013721																							-- (concept_id ����)


