import numpy as np
EXTRASTOPWORDS=[
'want',
]
MIXRANKDATA = [
    'connections',
    'org',
    'education',
    'title',
    'headline',
    'num_recommenders',
    'jobs_count',
    'languages',
    'person_id',
    'location_name',
    'slug',
    'certifications',
    'last_refresh',
    'country',
    'experience',
    'summary']

LINKEDDETAIL = [
    'connections',
    'org',
    'title',
    'headline',
    'num_recommenders',
    'jobs_count',
    'sourceType',
    'person_id',
    'location_name',
    'slug',
    'last_refresh',
    'country',
    'classificationLevel',
    'summary',
    'total_years_of_experience']

LINKEDDETAIL_DB = [
    'network_size',
    'organization',
    'current_title',
    'headline',
    'num_recommenders',
    'job_counts',
    'source_type',
    'source_id',
    'location_name',
    'slug',
    'source_refresh_date',
    'country',
    'classification_level',
    'profile_summary',
    'total_years_of_experience',
    'user_entity']

LANGUAGE = [
    'proficiency',
    'language']

LANGUAGE_DB = [
    'level',
    'name',
    'linkedin_detail_id']

EDUCATION = [
    'degree',
    'endDate',
    'grade',
    'schoolName',
    'field_of_study',
    'startDate']

EDUCATION_DB = [
    'degree',
    'end_date',
    'grade',
    'school_name',
    'field_of_study',
    'start_date',
    'linkedin_detail_id']

CERTIFICATION=[
    'title',
    'company_name',
    'verify_url',
    'date'
    ]

CERTIFICATION_DB=[
    'title',
    'company_name',
    'verify_url',
    'date',
    'linkedin_detail_id'
    ]

EXPERIENCE = [
    'end_date',
    'title',
    'company',
    'locality',
    'company_domain',
    'is_current',
    'start_date',
    'monthworked',
    'summary'
    ]

EXPERIENCE_DB = [
    'end_date',
    'title',
    'company',
    'locality',
    'summary',
    'company_domain',
    'is_current',
    'start_date',
    'month_worked',
    'linkedin_detail_id']
MATCHING_EDUCATION = [
    'degree',
    'school_name',
    'field_of_study',
    'linkedin_detail_id'
]
MATCHING_EXPERIENCE = [
    'company',
    'is_current',
    'month_worked',
    'summary',
    'title',
    'linkedin_detail_id'
]
MATCHING_LINKEDIN_DF = [
    'classification_level',
    'current_title',
    'headline',
    'job_counts',
    'network_size',
    'organization',
    'profile_summary',
    'skills',
    'experience',
    'education',
    'linkedin_url',
    'salary',
    'availability_hours_per_month',
    'response_time',
    'max_mentee_number',
    'total_years_of_experience',
    'count',
    'user_entity'
]

BACHELORS_DEGREE=[
    'Bachelor',
    '''Bachelor's''',
    'Bachelors',
    'bachelor',
    '''bachelor's''',
    'bachelors'
    'B.A.A.',
    'B.A.B.A',
    'B.A.Com.',
    'B.Acc.Sci.',
    'B.Acy',
    'B.A.E.',
    'B.A(Econ)',
    'B.A.J.',
    'B.A.M.',
    'B.A.O.M.',
    'B.A.P.S.Y.',
    'B.A.S.',
    'B.A.Sc.',
    'B.A.S.W.',
    'B.A.T.',
    'B.Ag',
    'B.App.Sc(IT)',
    'B.Arch.',
    'B.Avn.',
    'B.B.A.',
    'B.B.I.S.',
    'B.Bus.',
    'B.Bus.Sc.',
    'B.Ch.E.',
    'B.Com.',
    'B.Comp.',
    'B.Comp.Sc.',
    'B.Crim.',
    'B.C.A.',
    'B.C.E.',
    'B.C.J.',
    'B.Des.',
    'B.E.',
    'B.Ec.',
    'B.E.E.',
    'B.Eng.',
    'B.E.Sc.',
    'B.F.A.',
    'B.F&TV.',
    'B.G.S.',
    'B.H.S.',
    'B.I.B.E.',
    'B.In.Dsn.',
    'B.I.S.',
    'B.Kin.',
    'B.Sc.Kin.',
    'B.L.A.',
    'B.L.Arch.',
    'B.L.S.',
    'B.L.I.S.',
    'B.Lib.',
    'B.M.',
    'B.M.E',
    'B.M.O.S.',
    'B.M.S.',
    'B.Math',
    'B.Math.Sc.',
    'B.P.A.P.M.',
    'B.P.S.',
    'B.Phil.',
    'B.S.',
    'B.S.A.E.',
    'B.S.B.A.',
    'B.S.C.S.',
    'B.S.Chem.',
    'B.S.E.',
    'B.S.Ed.',
    'B.S.E.T.',
    'B.S.F.',
    'B.S.M.E.',
    'B.S.Micr.',
    'B.S.P.H.',
    'B.S.S.W.',
    'B.Sc.',
    'B.Sc(Econ)',
    'B.Sc(IT)',
    'B.Sc(Psych)',
    'B.Soc.Sc.',
    'B.T.S.',
    'B.Tech.',
    'B.U.R.P.',
    'B.A.',
    'B.Compt.',
    'B.Acc.',
    'B.J.',
    'B.A.Mus',
    'B.Comm.',
    'B.Econ.',
    'B.S.G.S.',
    'B.H.Sc.',
    'A.L.B.',
    'B.Mus.',
    'B.M.Ed.',
    'B.S.Eng.',
    'Ph.B.',
    'B.Plan.',
    'S.B.',
    'AB',
    'BAA',
    'BABA',
    'BACom',
    'BAccSci',
    'BAcy',
    'BAE',
    'BA(Econ)',
    'BAJ',
    'BAM',
    'BAOM',
    'BAPSY',
    'BAS',
    'BASc',
    'BASW',
    'BAT',
    'BAg',
    'BAppSc(IT)',
    'BArch',
    'BAvn',
    'BBA',
    'BBIS',
    'BBus',
    'BBusSc',
    'BChE',
    'BCom',
    'BComp',
    'BCompSc',
    'BCrim',
    'BCA',
    'BCE',
    'BCJ',
    'BDes',
    'BE',
    'BEc',
    'BEE',
    'BEng',
    'BESc',
    'BFA',
    'BF&TV',
    'BGS',
    'BHS',
    'BIBE',
    'BInDsn',
    'BIS',
    'BKin',
    'BScKin',
    'BLA',
    'BLArch',
    'BLS',
    'BLIS',
    'BLib',
    'BM',
    'BME',
    'BMOS',
    'BMS',
    'BMath',
    'BMathSc',
    'BPAPM',
    'BPS',
    'BPhil',
    'BS',
    'BSAE',
    'BSBA',
    'BSCS',
    'BSChem',
    'BSE',
    'BSEd',
    'BSET',
    'BSF',
    'BSME',
    'BSMicr',
    'BSPH',
    'BSSW',
    'BSc',
    'BSc(Econ)',
    'BSc(IT)',
    'BSc(Psych)',
    'BSocSc',
    'BTS',
    'BTech',
    'BURP',
    'BA',
    'BCompt',
    'BAcc',
    'BJ',
    'BAMus',
    'BComm',
    'BEcon',
    'BSGS',
    'BHSc',
    'ALB',
    'BMus',
    'BMEd',
    'BSEng',
    'PhB',
    'BPlan',
    'SB',
    'BAAS',
    'BAED',
    'BArch',
    'BAS',
    'BASW',
    'BBA',
    'BEd',
    'BEM',
    'BEnvD',
    'BFA',
    'BHS',
    'BIS',
    'BLS',
    'BMA',
    'BSB',
    'BSBME',
    'BSCE',
    'BSChE',
    'BSCIV',
    'BSCJ',
    'BSCLS',
    'BSCompEng',
    'BSEd/BSED',
    'BSEE',
    'BSElecEng',
    'Lisans Derecesi',
    'Mühendisliği',
    'Mühendisi',
    'Licentiate degree',
    "Engineer's Degree",
    'Undergraduate',
    'License',
    'license',
    'Lisans',
    'üniversitesi',
    'Üniversitesi',
    'University',
    'Institute',
    'Enstitüsü',
    'üniversitesi',
    'university',
    'institute',
    'enstitüsü',
    'Politechnika',
    'Univerza',
    'Politecnico',
    'Universität',
    'Harvard']

ASSOCIATE_DEGREE=[
    'Associate',
    'Associated',
    '''Associate's''',
    'Associates',
    'associate',
    'associates',
    '''associate's''',
    'A.A.',
    'A.A.-T',
    'A.A.A.',
    'A.A.B.',
    'A.A.S.',
    'A.A.T.',
    'A.B.A.',
    'A.B.S.',
    'A.E.E.T.',
    'A.E.',
    'A.E.',
    'A.E.S.',
    'A.E.T.',
    'A.F.',
    'A.F.A.',
    'A.G.',
    'A.I.T.',
    'A.O.S.',
    'A.O.T.',
    'A.P.E.',
    'A.P.S.',
    'A.S.',
    'A.S.-T',
    'A.S.',
    'A.S.P.T.orA.P.T',
    'A.T.',
    'A.A.T',
    'A.S.T',
    'A.',
    'AA',
    'AA-T',
    'AAA',
    'AAB',
    'AAS',
    'AAT',
    'ABA',
    'ABS',
    'AEET',
    'AE',
    'AE',
    'AES',
    'AET',
    'AF',
    'AFA',
    'AG',
    'AIT',
    'AOS',
    'AOT',
    'APE',
    'APS',
    'AS',
    'AS-T',
    'AS',
    'ASPT',
    'AT',
    'AAT',
    'AST',
    'APT',
    'A',
    'Önlisans',
    'Ön Lisans',
    'Yüksekokul',
    'önlisans',
    'ön lisans'

]
MASTER_DEGREE = [
    'master',
    'masters',
    'Master',
    'Masters',
    'M.A.',
    'M.Acc.',
    'M.Arch.',
    'M.Aqua.',
    'M.A.Ed.',
    'M.A.L.S.',
    'M.A.S.',
    'M.A.Sc.',
    'M.A.T.',
    'M.Bus.',
    'M.B.A.',
    'M.B.I.',
    'M.Chem.',
    'M.Com.',
    'M.Crim.',
    'M.C.A.',
    'M.C.D.',
    'M.C.F.',
    'M.C.J.',
    'M.C.P.',
    'M.C.S.',
    'M.C.T.',
    'M.Des.',
    'M.E.',
    'M.Econ.',
    'M.Ed.',
    'M.Ent.',
    'M.E.M.',
    'M.Fin.',
    'M.Fstry.',
    'M.F.A.',
    'M.F.E.',
    'M.H.',
    'M.H.A.',
    'M.H.S.',
    'M.I.Aff.',
    'M.I.B.',
    'M.I.L.R.',
    'M.I.S.',
    'M.I.S.M.',
    'M.I.T.',
    'M.L.A.',
    'M.L.Arch.',
    'M.L.I.S.',
    'M.Litt.',
    'M.M.',
    'M.Math.',
    'M.Mus.',
    'M.M.F.',
    'M.O.T.',
    'M.P.S.',
    'M.Phil.',
    'M.Phys.',
    'M.P.A.',
    'M.P.Aff.',
    'M.P.H.',
    'M.P.M.',
    'M.P.P.',
    'M.P.S.',
    'M.Poli.Sci.',
    'M.Q.F.',
    'M.R.',
    'M.R.E.D.',
    'M.S.',
    'M.S.C.J.',
    'M.S.C.S.',
    'M.S.Chem.',
    'M.S.E.',
    'M.S.Ed.',
    'M.S.Fin.',
    'M.S.F.S.',
    'M.S.H.R.D',
    'M.S.I.S.',
    'M.S.I.T',
    'M.S.L.',
    'M.S.M.',
    'M.S.M.I.S.',
    'M.S.M.Sci.',
    'M.S.Met.',
    'M.S.P.M.',
    'M.S.S.C.M',
    'M.S.Sc.',
    'M.S.T.',
    'M.St.',
    'M.Sw.E',
    'M.S.W.',
    'M.U.P.',
    'A.M.',
    'M.Acy.',
    'M.L.S.',
    'M.A.S.',
    'M.Comm.',
    'M.Design',
    'Ed.M.',
    'M.S.I.M',
    'S.C.M.',
    'H.R.D.',
    'M.S.I.S.M',
    'M.Sc.I.T.',
    'M.Sc.',
    'M.Sc.R.',
    'M.P.S.',
    'M.M.',
    'MA',
    'MAcc',
    'MArch',
    'MAqua',
    'MAEd',
    'MALS',
    'MAS',
    'MASc',
    'MAT',
    'MBus',
    'MBA',
    'MBI',
    'MChem',
    'MCom',
    'MCrim',
    'MCA',
    'MCD',
    'MCF',
    'MCJ',
    'MCP',
    'MCS',
    'MCT',
    'MDes',
    'ME',
    'MEcon',
    'MEd',
    'MEnt',
    'MEM',
    'MFin',
    'MFstry',
    'MFA',
    'MFE',
    'MH',
    'MHA',
    'MHS',
    'MIAff',
    'MIB',
    'MILR',
    'MIS',
    'MISM',
    'MIT',
    'MLA',
    'MLArch',
    'MLIS',
    'MLitt',
    'MM',
    'MMath',
    'MMus',
    'MMF',
    'MOT',
    'MPS',
    'MPhil',
    'MPhys',
    'MPA',
    'MPAff',
    'MPH',
    'MPM',
    'MPP',
    'MPS',
    'MPoliSci',
    'MQF',
    'MR',
    'MRED',
    'MS',
    'MSCJ',
    'MSCS',
    'MSChem',
    'MSE',
    'MSEd',
    'MSFin',
    'MSFS',
    'MSHRD',
    'MSIS',
    'MSIT',
    'MSL',
    'MSM',
    'MSMIS',
    'MSMSci',
    'MSMet',
    'MSPM',
    'MSSCM',
    'MSSc',
    'MST',
    'MSt',
    'MSwE',
    'MSW',
    'MUP',
    'AM',
    'MAcy',
    'MLS',
    'MAS',
    'MComm',
    'MDesign',
    'EdM',
    'MSIM',
    'SCM',
    'HRD',
    'MSISM',
    'MScIT',
    'MSc',
    'MScR',
    'MPS',
    'MM',
    'Yüksek',
    'Graduate'
]
DOCTORATE_DEGREE = [
    'Doctor',
    'doctor',
    'Doctorate',
    'doctorate',
    'Doctoral',
    'doctoral',
    'Au.D.',
    'Art.D.',
    'D.Arch.',
    'D.A.T.',
    'D.A.S.',
    'D.B.A.',
    'D.C.',
    'D.Chem.',
    'D.Crim.',
    'D.C.J.',
    'D.Des.',
    'D.Ed.',
    'D.Eng.',
    'D.Env.',
    'D.F.',
    'D.F.A.',
    'D.G.S.',
    'D.H.S.',
    'D.I.T.',
    'D.L.S.',
    'D.M.',
    'D.M.A.',
    'D.M.L.',
    'D.P.A.',
    'D.P.E.',
    'D.P.H.',
    'D.P.S.',
    'D.R.',
    'D.Sc.',
    'D.Sc.H.',
    'D.S.W.',
    'L.H.D.',
    'Mus.D.',
    'Ph.D.',
    'S.D',
    'S.Sc.D.',
    'D.A.',
    'Sc.D.',
    'D.M.',
    'Sc.D.',
    'D.Rec.',
    'Ed.D',
    'AuD',
    'ArtD',
    'DArch',
    'DAT',
    'DAS',
    'DBA',
    'DC',
    'DChem',
    'DCrim',
    'DCJ',
    'DDes',
    'DEd',
    'DEng',
    'DEnv',
    'DF',
    'DFA',
    'DGS',
    'DHS',
    'DIT',
    'DLS',
    'DM',
    'DMA',
    'DML',
    'DPA',
    'DPE',
    'DPH',
    'DPS',
    'DR',
    'DSc',
    'DScH',
    'DSW',
    'LHD',
    'MusD',
    'PhD',
    'SD',
    'SScD',
    'DA',
    'ScD',
    'DM',
    'ScD',
    'DRec',
    'EdD',

]
IT_SKILLS = [
    'SQL', 'Python', 'Azure', 'AWS', 'Data Analysis', 'JavaScript', 'Software Development Lifecycle', 'Java', 'DevOps', 'Continuous Integration', 'C#', 'SAP', 'Scrum', 'Machine Learning', 'HTML', 'Linux', 'APIs', 'CD', 'React', 'AI', 'CSS', 'Salesforce', 'CRM', 'Power BI', 'SQL Server', 'Business Intelligence', 'Jira', 'Scripting Language', '.NET', 'Microservices', 'Docker', 'Kubernetes', 'Git', 'AutoCAD', 'C++', 'UX', 'SharePoint', 'Data Science', 'SaaS', 'UI', 'Angular', 'Photoshop', 'GCP', 'Active Directory', 'CAD', 'TDD', 'SEO', 'Node.js', 'Google Analytics', 'Automated Testing', 'Tableau', 'TypeScript', 'Big Data', 'Algorithms', 'Firewalls', 'Jenkins', 'R', 'Terraform', '.NET Core', 'Cloud Platforms', 'MySQL', 'Illustrator', 'Kanban', 'Version Control', 'Database Design', 'InDesign', 'C', 'Data Warehouse', 'PowerShell', 'PHP', 'Cisco', 'Unit Testing', 'VMware', 'Virtualization', 'PRINCE2', 'BDD', 'SOLID', 'DNS', 'Vue.js', 'Revit', 'Data Visualization', 'Graphic design', 'Confluence', 'MongoDB', 'Network Switches', 'Web Services', 'Data Modeling', 'Code Reviews', 'Pivot Tables', 'Design Patterns', 'NoSQL', 'Unix', 'ETL', 'Go', 'MS Project', 'Ansible', 'Kafka', 'GitHub', 'Android', 'RESTful APIs', 'Microsoft Azure', 'React.js', 'Cloud Computing', 'Software Development'
]