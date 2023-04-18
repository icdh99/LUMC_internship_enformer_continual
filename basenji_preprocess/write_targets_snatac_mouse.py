lines = [['index','identifier','file','clip', 'scale', 'sum_stat','description']]

lines.append(['0', 'L2-3_IT', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L2-3_IT.bw', '32', '2', 'mean', 'ATAC:L2-3_IT']) 
lines.append(['1', 'L5_ET', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L5_ET.bw', '32', '2', 'mean', 'ATAC:L5_ET']) 
lines.append(['2', 'L5_IT_Rspo1', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L5_IT_Rspo1.bw', '32', '2', 'mean', 'ATAC:L5_IT_Rspo1']) 
lines.append(['3', 'L5_IT_Rspo2', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L5_IT_Rspo2.bw', '32', '2', 'mean', 'ATAC:L5_IT_Rspo2']) 
lines.append(['4', 'L5_IT_S100b', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L5_IT_S100b.bw', '32', '2', 'mean', 'ATAC:L5_IT_S100b']) 
lines.append(['5', 'L6_CT', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L6_CT.bw', '32', '2', 'mean', 'ATAC:L6_CT']) 
lines.append(['6', 'L6_IT', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L6_IT.bw', '32', '2', 'mean', 'ATAC:L6_IT']) 
lines.append(['7', 'L6_NP', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/L6_NP.bw', '32', '2', 'mean', 'ATAC:L6_NP']) 
lines.append(['8', 'Lamp5', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/Lamp5.bw', '32', '2', 'mean', 'ATAC:Lamp5']) 
lines.append(['9', 'Pvalb_Calb1', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/Pvalb_Calb1.bw', '32', '2', 'mean', 'ATAC:Pvalb_Calb1']) 
lines.append(['10', 'Pvalb_Reln', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/Pvalb_Reln.bw', '32', '2', 'mean', 'ATAC:Pvalb_Reln']) 
lines.append(['11', 'Sst', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/Sst.bw', '32', '2', 'mean', 'ATAC:Sst']) 
lines.append(['12', 'Vip', '/exports/humgen/idenhond/Analysis_EckerRen_Mouse_MOp_methylation_ATAC/data/EckerRen_Mouse_MOp_methylation_ATAC/bigwig/EpigenomeClustering/ATAC/Vip.bw', '32', '2', 'mean', 'ATAC:Vip']) 

samples_out = open('/exports/humgen/idenhond/data/basenji_preprocess/targets_snatac_mouse.txt', 'w')
for line in lines:
    print('\t'.join(line), file=samples_out)
samples_out.close()