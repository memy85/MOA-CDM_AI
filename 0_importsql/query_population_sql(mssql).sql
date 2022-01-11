--To generate only_IO_population
--Ignore drop error

--1. lung
	--drop table CDM.dbo.temp_population
	select * into CDM.dbo.temp_population
	from
		(select person_id as subject_id, year_of_birth, gender_source_value from CDM.dbo.person) as person

		inner join

		(select person_id, condition_concept_id, condition_start_date
			from CDM.dbo.condition_occurrence
			where condition_concept_id in (256646,257503,258375,261236,443388,4092217,4094876,4151250,4157333,4157454,4311499) --lung concept_id
		) as lung

		on person.subject_id = lung.person_id


--2. only IO after lung diagnosis
	--drop table CDM.dbo.population_only_IO
	select * into CDM.dbo.population_only_IO
	from
	(
		select subject_id, min(drug_exposure_start_date) as cohort_start_date, 
				(YEAR(GETDATE() )-year_of_birth  ) as age, 
				gender_source_value, person_id
		from
	
			(select * from CDM.dbo.temp_population) as p
			inner join	
			(select person_id as d_person_id, drug_exposure_start_date	
				from CDM.dbo.drug_exposure 
				where drug_concept_id in (1301125,1314865,1597773,1742287,2718527,19077652,35603172,40141719,36889829,40080069,44817887,
									708298,1154186,2213440,1000560,1114122,2718836,19081294,19135374,752211,1343916,21062401,
									36890262,40072348,40227542,40929934,2213483,2718805,40099452,1518606,19110450,40102878,
									40169889,40173129,40243389,1367571,2718321,19081616,40042274,40836918,41302979) --IO concept_id
			) as ICI
		
			on
			p.subject_id = ICI.d_person_id and p.condition_start_date <= ICI.drug_exposure_start_date
		group by subject_id, year_of_birth, gender_source_value, person_id
	) as t

--3. without Chemo
	delete from aa
	from
	
		(select * from CDM.dbo.population_only_IO) as aa
		inner join
		(select person_id, drug_exposure_start_date
		from CDM.dbo.drug_exposure
		where drug_concept_id in (792499,793797,903643,905078,941052,955632,988447,1118084,1301267,1304919,1305058,1308290,1309161,1309188,1310317,1311078,
								1311409,1311443,1311799,1314865,1314924,1315942,1326481,1329241,1333357,1333379,1336825,1337620,1337651,1338512,1341149,1343346,
								1344354,1344905,1349025,1350066,1350504,1355509,1355795,1356361,1361191,1367268,1368823,1377141,1378382,1378509,1381253,1389036,1389888,
								1390051,1391846,1394337,1395557,1396423,1397599,1436650,1437379,1560123,1593861,1718850,19002912,19006880,19008264,19008336,19009165,19012543,
								19012585,19015523,19017810,19024728,19025348,19031224,19038536,19042545,19046625,19051642,19054821,19054825,19056756,19057483,19069046,19078097,
								19078187,19093366,19101677,19104221,19125635,19135793,19136210,19136750,19137385,35201068,35603017,35604205,35606214,40166461,40168385,40222431,40230712,
								40244266,42709321,42873638,42903942,44816310,45776670,45776944,45892579,46221435,46275491) --Chemo_concept_id
		) as bb
		on
		aa.person_id=bb.person_id and bb.drug_exposure_start_date >= aa.cohort_start_date


--4. 확인
	select * from CDM.dbo.population_only_IO