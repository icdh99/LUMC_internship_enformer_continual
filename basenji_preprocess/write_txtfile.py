# conda activate basenji

# lines = [['index','identifier','file','clip','sum_stat','description']]
# lines.append(['0', 'CNhs11760', '/exports/humgen/idenhond/data/basenji_preprocess/CNhs11760.bw', '384', 'sum', 'aorta'])
# lines.append(['1', 'CNhs12843', '/exports/humgen/idenhond/data/basenji_preprocess/CNhs12843.bw', '384', 'sum', 'artery'])
# lines.append(['2', 'CNhs12856', '/exports/humgen/idenhond/data/basenji_preprocess/CNhs12856.bw', '384', 'sum', 'pulmonic_valve'])

# samples_out = open('/exports/humgen/idenhond/data/basenji_preprocess/heart_wigs.txt', 'w')
# for line in lines:
#     print('\t'.join(line), file=samples_out)
# samples_out.close()

lines = [['index','identifier','file','clip', 'scale', 'sum_stat','description']]
lines.append(['0', 'ENCFF601VTB', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF601VTB.bw', '32', '2', 'mean', 'brain_tissue_female_embryo'])  # histone chip encode
lines.append(['1', 'ENCFF914YXU', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF914YXU.bw', '32', '2', 'mean', 'liver_tissue'])   # histone chip encode
lines.append(['2', 'ENCFF833POA', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF833POA.bw', '32', '2', 'mean', 'DNASE:cerebellum male adult (27 years) and male adult (35 years)'])
lines.append(['3', 'ENCFF828RQS', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF828RQS.bw', '32', '2', 'mean', 'CHIP:H3K9me3:stomach smooth muscle female adult (84 years)'])
lines.append(['4', 'ENCFF003HJB', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF003HJB.bw', '32', '2', 'mean', 'CHIP:CEBPB:HepG2'])
lines.append(['5', 'ENCFF039UUS', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF039UUS.bw', '32', '2', 'mean', 'Homo sapiens K562 genetically modified (insertion) using CRISPR targeting H. sapiens TEAD1'])
lines.append(['6', 'ENCFF967MGL', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF967MGL.bw', '32', '2', 'mean', 'Homo sapiens right lobe of liver tissue female adult (47 years)'])
lines.append(['7', 'ENCFF956MZC', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF956MZC.bw', '32', '2', 'mean', 'Homo sapiens natural killer cell male adult (33 years)'])
lines.append(['8', 'ENCFF730XOV', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF730XOV.bw', '32', '2', 'mean', 'Homo sapiens middle frontal area 46 tissue male adult (83 years)'])
lines.append(['9', 'ENCFF924IJQ', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF924IJQ.bw', '32', '2', 'mean', 'Homo sapiens middle frontal area 46 tissue female adult (79 years)'])
lines.append(['10', 'ENCFF417AGZ', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF417AGZ.bw', '32', '2', 'mean', 'Homo sapiens middle frontal area 46 tissue female adult (82 years)'])
lines.append(['11', 'ENCFF949NAK', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF949NAK.bw', '32', '2', 'mean', 'Homo sapiens WTC11 genetically modified (insertion) using CRISPR targeting H. sapiens USF1'])
lines.append(['12', 'ENCFF194XNN', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF194XNN.bw', '32', '2', 'mean', 'Homo sapiens with Alzheimers disease; middle frontal area 46 tissue female adult (90 or above years)'])
lines.append(['13', 'ENCFF148LUF', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF148LUF.bw', '32', '2', 'mean', 'Homo sapiens ovary tissue female adult (41 years)'])
lines.append(['14', 'ENCFF446DQS', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF446DQS.bw', '32', '2', 'mean', 'Homo sapiens heart right ventricle tissue male adult (54 years)'])
lines.append(['15', 'ENCFF873YYI', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF873YYI.bw', '32', '2', 'mean', 'H3K27me3 ChIP-Seq on untreated BLaER1 cell line.'])
lines.append(['16', 'ENCFF515FSI', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF515FSI.bw', '32', '2', 'mean', 'Homo sapiens BLaER1 48 hours after the sample was treated with 100 nM 17β-estradiol, 10 ng/mL Interleukin-3, 10 ng/mL CSF1'])
lines.append(['17', 'ENCFF781NJP', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF781NJP.bw', '32', '2', 'mean', 'Homo sapiens HepG2 genetically modified (insertion) using CRISPR targeting H. sapiens BORCS8'])
lines.append(['18', 'ENCFF700IQJ', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF700IQJ.bw', '32', '2', 'mean', 'Homo sapiens heart left ventricle tissue male adult (69 years)'])
lines.append(['19', 'ENCFF107RAU', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF107RAU.bw', '32', '2', 'mean', 'Homo sapiens A673'])
lines.append(['20', 'ENCFF532YBF', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF532YBF.bw', '32', '2', 'mean', 'Homo sapiens naive B cell female adult (39 years)'])
lines.append(['21', 'ENCFF932AQG', '/exports/humgen/idenhond/data/basenji_preprocess/ENCFF932AQG.bw', '32', '2', 'mean', 'Homo sapiens immature natural killer cell'])


samples_out = open('/exports/humgen/idenhond/data/basenji_preprocess/target_march24.txt', 'w')
for line in lines:
    print('\t'.join(line), file=samples_out)
samples_out.close()