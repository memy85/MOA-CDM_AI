--cohort table join person
	drop table CDM.dbo.population_only_IO
	select * into CDM.dbo.population_only_IO
	from
	(
		select * 
		from
		(
			select subject_id, cohort_start_date
			 from WEBAPI.dbo.cohort
			 where cohort_definition_id = 106 ---------------------change
		) as a
		inner join
		(
			select ( date_part('year', current_date) - year_of_birth  ) as age, gender_source_value, person_id
			 from CDM.dbo.person
			 ) as b
		on
		a.subject_id=b.person_id
	) as t

--확인
	select * from CDM.dbo.population_only_IO