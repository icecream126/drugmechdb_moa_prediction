Contents of Data
----------------


NOTE: The path to this dir can be changed in `drugmechcf.utils.projconfig.ProjectConfig.get_input_data_dir()`.



Counterfactuals/	... Drug Mechanisms Counterfactual samples
		See MoaData/README.txt


The following data to be downloaded by the user.


DrugMechDB/	... Download from: https://github.com/SuLab/DrugMechDB

	Raw data files:

	indication_paths.yaml
		Src: https://github.com/SuLab/DrugMechDB/blob/main/indication_paths.yaml
		version tested: July 10, 2023

	deprecated_ids.txt
		Src: https://github.com/SuLab/DrugMechDB/blob/main/utils/deprecated_ids.txt
		version tested: Apr 4, 2023

	To build and cache, see:
		`drugmechcf.data.drugmechdb.load_drugmechdb()`



MONDO/		... Download from: https://mondo.monarchinitiative.org
		    version tested: June 19, 2024

	Raw data files:

	mondo.obo

	To build and cache, see:
		`drugmechcf.data.mondo.load_mondo()`



PrimeKG/
	Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
	version tested: April 25, 2022

	Raw data files:

	disease_features.csv
		textual descriptions of diseases

	drug_features.csv
		textual descriptions of drugs

	edges.csv.gz

	kg.csv.gz
		Main PrimeKG data. Each row is an edge. First row is header.

	To build and cache, see:
		`drugmechcf.data.primekg.load_primekg()`
