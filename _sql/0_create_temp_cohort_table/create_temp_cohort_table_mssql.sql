IF OBJECT_ID('@cohort_database_schema.@cohort_table', 'U') IS NULL
	CREATE TABLE @cohort_database_schema.@cohort_table (
	  cohort_definition_id INT,
	  subject_id BIGINT,
	  cohort_start_date DATE,
	  cohort_end_date DATE
	);