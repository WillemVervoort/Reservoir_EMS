import os
import sys
import zipfile
from rasterio.merge import merge
import rasterio

def extract_files(path):
    '''Extracts and create list of extracted tiff files'''
    if 'Extract' not in os.listdir(path):
        os.mkdir(os.path.join(path, 'Extract'))
    files = [n for n in os.listdir(path) if (n.endswith('.zip'))]
    for n in files:
        with zipfile.ZipFile(os.path.join(path,n), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(path, 'Extract'))
    tiffs = [n for n in os.listdir(os.path.join(path, 'Extract')) if (n.endswith('.tif')) | (n.endswith('.asc'))]
    return tiffs


def to_mosaic(path, tif_list):
    '''mosaic file based on list of tiff files'''
    src_files_to_mosaic = []
    for fp in tif_list:
        src = rasterio.open(os.path.join(path, 'Extract', fp))
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "dtype": 'float32',
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans})
    with rasterio.open(os.path.join(path, 'Extract', 'mosaic'), "w", **out_meta) as dest:
        dest.write(mosaic)

path = str(sys.argv[1]).strip()
files = extract_files(path)
to_mosaic(path, files)

import h5py
import numpy as np
filename = "ATL13_20201106043223_06570901_003_01.h5"

f1 = h5py.File(filename, "r")
list(f1.keys())
x1 = f1['gt1r']
df1 = np.array(x1.values)
df1.shape
with h5py.File(filename, "r") as f:
    print(f)
    # List all groups
    print("Keys: %s" % f.keys())
    print(f.keys())
    print(f['gt1l'])
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
def read_hdf5(path):

    weights = {}

    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                print(f[key].name)
                weights[f[key].name] = f[key].value
    return weights

read_hdf5(filename)

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scikit_posthocs as sp
%matplotlib inline

df = pd.read_csv('dif_duration.csv')
df.groupby('clim').mean()
df['clim'].unique()
input = [df[df['clim'] == n]['b1'].values for n in df['clim'].unique()]
sp.posthoc_dunn(input3, p_adjust = 'holm')
help(sp.posthoc_dunn)
0e0
plt.hist(1/np.log(input[0]), bins=50)


stats.shapiro(1/np.log(input[0]))
stats.kruskal(input3[0], input3[1], input3[2], input3[3], input3[4])
stats.median_test(input3[0], input3[1], input3[2], input3[3], input3[4])

stats.levene(input[0], input[1], input[2], input[3], input[4])
help(sp.posthoc_dun)
labels = ['Tropical', 'Dry', 'Temperate', 'Continental', 'Polar']
plt.boxplot((input[0], input[1], input[2], input[3], input[4]), labels=labels, showfliers=False)

df2 = pd.read_csv('dif_severity.csv')
df2.groupby('clim').mean()
df2['clim'].unique()
input2 = [df2[df2['clim'] == n]['b1'].values for n in df2['clim'].unique()]

df3 = pd.read_csv('SSI_dif_intensity.csv')
df3.groupby('clim').mean()
df3['clim'].unique()
input3 = [df3[df3['clim'] == n]['b1'].values for n in df3['clim'].unique()]


dfx = pd.read_csv('duration_SPEI.csv')
dfx.groupby('clim').mean()
dfx['clim'].unique()
inputx = [dfx[dfx['clim'] == n]['MASK'].values for n in dfx['clim'].unique()]

dfx_1 = pd.read_csv('SPEI_severity.csv')
dfx_1.groupby('clim').mean()
dfx_1['clim'].unique()
inputx_1 = [dfx_1[dfx_1['clim'] == n]['SPEI'].values for n in dfx_1['clim'].unique()]

dfx_2 = pd.read_csv('SPEI_intensity.csv')
dfx_2.groupby('clim').mean()
dfx_2['clim'].unique()
inputx_2 = [dfx_2[dfx_2['clim'] == n]['SPEI'].values for n in dfx_2['clim'].unique()]

dfz = pd.read_csv('SVI_duration.csv')
dfz.groupby('clim').mean()
dfz['clim'].unique()
inputz = [dfz[dfz['clim'] == n]['MASK'].values for n in dfz['clim'].unique()]

dfz_1 = pd.read_csv('SVI_severity.csv')
dfz_1.groupby('clim').mean()
dfz_1['clim'].unique()
inputz_1 = [dfz_1[dfz_1['clim'] == n]['SVI'].values for n in dfz_1['clim'].unique()]

dfz_2 = pd.read_csv('SVI_intensity.csv')
dfz_2.groupby('clim').mean()
dfz_2['clim'].unique()
inputz_2 = [dfz_2[dfz_2['clim'] == n]['SVI'].values for n in dfz_2['clim'].unique()]

dfy = pd.read_csv('SSI_duration.csv')
dfy.groupby('clim').mean()
dfy['clim'].unique()
inputy = [dfy[dfy['clim'] == n]['MASK'].values for n in dfy['clim'].unique()]

dfy_1 = pd.read_csv('SSI_severity.csv')
dfy_1.groupby('clim').mean()
dfy_1['clim'].unique()
inputy_1 = [dfy_1[dfy_1['clim'] == n]['SPEI'].values for n in dfy_1['clim'].unique()]

dfy_2 = pd.read_csv('SSI_intensity.csv')
dfy_2.groupby('clim').mean()
dfy_2['clim'].unique()
inputy_2 = [dfy_2[dfy_2['clim'] == n]['SPEI'].values for n in dfy_2['clim'].unique()]

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)

medianprops = dict(linewidth=0)
colors = ['darkgreen', 'red', 'orange', 'blue', 'grey']
ax1.boxplot(input, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
parts = ax1.violinplot(input, showmedians=True, showextrema=False)
parts['cmedians'].set_color('black')
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    # pc.set_edgecolor('black')
    pc.set_alpha(0.7)
# ax1.violinplot(input, showmedians=True, showextrema=False)
ax1.set_ylim(-5, 15)
ax1.axhline(0, ls='--', color='black', alpha=0.3)
ax1.set_ylabel('Duration', fontsize=11, labelpad=24)

ax2.boxplot(input2, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
# ax2.violinplot(input2, showmedians=True, showextrema=False)
parts2 = ax2.violinplot(input2, showmedians=True, showextrema=False)
parts2['cmedians'].set_color('black')
for i, pc in enumerate(parts2['bodies']):
    pc.set_facecolor(colors[i])
    # pc.set_edgecolor('black')
    pc.set_alpha(0.7)
ax2.set_ylim(-4, 12)
ax2.axhline(0, ls='--', color='black', alpha=0.3)
ax2.set_ylabel('Severity', fontsize=11, labelpad=24)

ax3.boxplot(input3, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
# ax3.violinplot(input3, showmedians=True, showextrema=False)
parts3 = ax3.violinplot(input3, showmedians=True, showextrema=False)
parts3['cmedians'].set_color('black')
for i, pc in enumerate(parts3['bodies']):
    pc.set_facecolor(colors[i])
    # pc.set_edgecolor('black')
    pc.set_alpha(0.7)
ax3.set_ylim(-0.4, 0.4)
ax3.axhline(0, ls='--', color='black', alpha=0.3)
ax3.set_ylabel('Intensity', fontsize=11, labelpad=1)
ax3.tick_params(axis='x', labelsize=11)

fig.text(-0.025, 0.3, 'Difference SRI - SPEI', rotation=90, fontsize=12)
fig.savefig('dif_SRI_violin.png', dpi=300, bbox_inches='tight')
help(fig.savefig)
parts['bodies'][0]


fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9] = plt.subplots(9, 1, figsize=(6,10), sharex=True)

medianprops = dict(linewidth=0)
ax1.boxplot(inputx, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
ax1.violinplot(inputx, showmedians=True, showextrema=False)
ax1.set_ylim(2, 8)
ax1.set_ylabel('Duration', fontsize=11, labelpad=2.5)

ax2.boxplot(inputx_1, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
ax2.violinplot(inputx_1, showmedians=True, showextrema=False)
ax2.set_ylim(2, 6)
ax2.set_ylabel('Severity', fontsize=11, labelpad=12)

ax3.boxplot(inputx_2, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
ax3.violinplot(inputx_2, showmedians=True, showextrema=False)
ax3.set_ylim(0.4, 0.8)
ax3.set_ylabel('Intensity', fontsize=11, labelpad=2.5)
ax3.tick_params(axis='x', labelsize=11)

ax4.boxplot(inputz, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
p4 = ax4.violinplot(inputz, showmedians=True, showextrema=False)
for pc in p4['bodies']:
    pc.set_facecolor('green')
    pc.set_alpha(0.5)
ax4.set_ylim(0, 12)
ax4.set_ylabel('Duration', fontsize=11, labelpad=5)

ax5.boxplot(inputz_1, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
p5 = ax5.violinplot(inputz_1, showmedians=True, showextrema=False)
for pc in p5['bodies']:
    pc.set_facecolor('green')
    pc.set_alpha(0.5)
ax5.set_ylim(0, 9)
ax5.set_ylabel('Severity', fontsize=11, labelpad=11)

ax6.boxplot(inputz_2, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
p6 = ax6.violinplot(inputz_2, showmedians=True, showextrema=False)
for pc in p6['bodies']:
    pc.set_facecolor('green')
    pc.set_alpha(0.5)
ax6.set_ylim(-0.2, 1.2)
ax6.set_ylabel('Intensity', fontsize=11, labelpad=11)
ax6.tick_params(axis='x', labelsize=11)

ax7.boxplot(inputy, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
p7 = ax7.violinplot(inputy, showmedians=True, showextrema=False)
for pc in p7['bodies']:
    pc.set_facecolor('orange')
    pc.set_alpha(0.5)
ax7.set_ylim(0, 17)
ax7.set_ylabel('Duration', fontsize=11, labelpad=5.5)

ax8.boxplot(inputy_1, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
p8 = ax8.violinplot(inputy_1, showmedians=True, showextrema=False)
for pc in p8['bodies']:
    pc.set_facecolor('orange')
    pc.set_alpha(0.5)
ax8.set_ylim(0, 13)
ax8.set_ylabel('Severity', fontsize=11, labelpad=5)

ax9.boxplot(inputy_2, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
p9 = ax9.violinplot(inputy_2, showmedians=True, showextrema=False)
for pc in p9['bodies']:
    pc.set_facecolor('orange')
    pc.set_alpha(0.5)
ax9.set_ylim(0.2, 1)
ax9.set_ylabel('Intensity', fontsize=11, labelpad=2)
ax9.tick_params(axis='x', labelsize=11)

fig.text(0, 0.78, '3-month SPEI', rotation=90, fontsize=12)
fig.text(0, 0.46, '3-month SVI', rotation=90, fontsize=12)
fig.text(0, 0.14, '3-month SRI', rotation=90, fontsize=12)
fig.tight_layout()
fig.savefig('indices_violin.png', dpi=300)

[np.std(n) for n in input]
[np.mean(n) for n in input]


dfz = pd.read_csv('spei_smi_lag.csv')
dfz.groupby('clim').median()
dfz['clim'].unique()
inputz = [dfz[dfz['clim'] == n]['b1'].values for n in dfz['clim'].unique()]

dfz_1 = pd.read_csv('spei_svi_lag.csv')
dfz_1.groupby('clim').median()
dfz_1['clim'].unique()
inputz_1 = [dfz_1[dfz_1['clim'] == n]['b1'].values for n in dfz_1['clim'].unique()]
[np.nanmedian(n) for n in inputz_1]
np.percentile(inputz_1[0], [50])

dfz_2 = pd.read_csv('SSI_lag.csv')
dfz_2.groupby('clim').median()
dfz_2['clim'].unique()
inputz_2 = [dfz_2[dfz_2['clim'] == n]['b1'].values for n in dfz_2['clim'].unique()]

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)



medianprops = dict(linewidth=0)
colors = ['darkgreen', 'red', 'orange', 'blue', 'grey']
ax1.boxplot(inputz, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
parts = ax1.violinplot(inputz, showmedians=True, showextrema=False)
parts['cmedians'].set_color('black')
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    # pc.set_edgecolor('black')
    pc.set_alpha(0.7)
# ax1.violinplot(input, showmedians=True, showextrema=False)
ax1.set_ylim(0, 9)
ax1.axhline(0, ls='--', color='black', alpha=0.3)
ax1.set_ylabel('SPEI-SMI', fontsize=11, labelpad=10)

ax2.boxplot(inputz_1, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
# ax2.violinplot(input2, showmedians=True, showextrema=False)
parts2 = ax2.violinplot(inputz_1, showmedians=True, showextrema=False)
parts2['cmedians'].set_color('black')
for i, pc in enumerate(parts2['bodies']):
    pc.set_facecolor(colors[i])
    # pc.set_edgecolor('black')
    pc.set_alpha(0.7)
ax2.set_ylim(0, 9)
ax2.axhline(0, ls='--', color='black', alpha=0.3)
ax2.set_ylabel('SPEI-SVI', fontsize=11, labelpad=10)

ax3.boxplot(inputz_2, showfliers=False, widths=[0.05, 0.05, 0.05, 0.05, 0.05], patch_artist=True, labels=labels, medianprops=medianprops)
# ax2.violinplot(input2, showmedians=True, showextrema=False)
parts3 = ax3.violinplot(inputz_2, showmedians=True, showextrema=False)
parts3['cmedians'].set_color('black')
for i, pc in enumerate(parts3['bodies']):
    pc.set_facecolor(colors[i])
    # pc.set_edgecolor('black')
    pc.set_alpha(0.7)
ax3.set_ylim(0, 9)
ax3.axhline(0, ls='--', color='black', alpha=0.3)
ax3.set_ylabel('SPEI-SRI', fontsize=11, labelpad=10)


fig.text(0, 0.3, 'Lagged response (months)', rotation=90, fontsize=12)
fig.savefig('lags_violin.png', dpi=300)


import pymannkendall
df = pd.read_csv('dischargeAli.csv')
X = df[df.columns[0]][df[df.columns[0]] > 1970].values
X = sm.add_constant(X)
y = df[df.columns[1]][df[df.columns[0]] > 1970].values

model = sm.OLS(y, X, missing='drop')
results = model.fit()
print(results.summary())

plt.plot(df[df.columns[0]], df[df.columns[1]], color='blue', label='T20')
plt.plot(df[df.columns[0]], df[df.columns[2]], color='red', label='E03')
plt.plot(range(1970, 2006), [(n*-15.2224 + 3.08e04) for n in range(1970, 2006)], color='blue', ls='--', label='trend T20')
plt.ylabel('River discharge (m$^3$ s$^{-1}$)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.legend(fontsize=12, loc=1)
plt.tight_layout()
plt.savefig('Ali_rivers.png', dpi=300)
