"""Check every A2 PNG heatmap/mask cell directly against saved raw NPZ.

Cell-center method reused from the previously accepted fixed-source verifier,
9ab3d0ea5d32db00e278f74c599a1943885453db:
manifests/grid-shuffle-20261017/results-audit/verify_attention_exports.py
No old audit is rerun; no figure is redrawn or edited.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from matplotlib import colormaps
from numeric_core import ident,load,read_bundle
ROOT=Path(__file__).resolve().parent

def verify_png(path,local,final,mask,product):
    with Image.open(path) as im:
        im.load();assert im.size==(product['width'],product['height'])==(2880,1280)
        rgb=np.asarray(im.convert('RGB'))
    black=rgb.max(axis=2)<50;groups=[]
    for y in range(rgb.shape[0]):
        edges=np.flatnonzero(np.diff(np.r_[False,black[y],False]))
        runs=[(left,right) for left,right in zip(edges[::2],edges[1::2]) if right-left>rgb.shape[1]*0.6]
        if not runs:continue
        left,right=max(runs,key=lambda pair:pair[1]-pair[0]);row=(y,int(left),int(right))
        if not groups or y>groups[-1][-1][0]+1:groups.append([row])
        else:groups[-1].append(row)
    assert len(groups)>=5,path
    scale=float(max(local.max(),final.max()));assert product['scale_min']==0 and product['scale_max']==scale
    maximum_error=0
    for panel,matrix in enumerate((local,final)):
        top,bottom=groups[2*panel][-1],groups[2*panel+1][0];left,right=top[1:]
        assert bottom[1:]==(left,right)
        ys=np.floor(top[0]+(np.arange(4)+0.5)*(bottom[0]-top[0])/4).astype(int)
        xs=np.floor(left+(np.arange(matrix.shape[1])+0.5)*(right-left)/matrix.shape[1]).astype(int)
        actual=rgb[ys[:,None],xs[None,:]];expected=colormaps['YlOrRd'](matrix/scale,bytes=True)[...,:3]
        error=int(np.abs(actual.astype(int)-expected.astype(int)).max());assert error<=1,(path,panel,error)
        maximum_error=max(maximum_error,error)
    top,bottom=groups[4][0],groups[-1][-1]
    assert top[0]<bottom[0]
    xs=np.floor(top[1]+(np.arange(len(mask))+0.5)*(top[2]-top[1])/len(mask)).astype(int)
    actual_mask=rgb[(top[0]+bottom[0])//2,xs];expected_mask=np.repeat(np.where(mask,0,255)[:,None],3,axis=1)
    assert np.abs(actual_mask.astype(int)-expected_mask).max()<=1,path
    return {'heatmap_cells_checked':int(2*local.size),'mask_cells_checked':int(len(mask)),
            'maximum_RGB_difference_0_to_255':maximum_error}

def main():
    conditions={};allimages=[]
    for n in [0,3,4]:
        directory=ROOT/'terminal_outputs/raw/jobs'/f'ga-20261018-sportsa2-attention-{n}-fd-recovery-v1/output'
        heatmap=mask_cells=maximum=0
        for batch in range(4):
            p=directory/f'batch-{batch:02d}';bundle=read_bundle(p/'bundle/record.json');receipt=load(p/'figures/plot_receipt.json')
            for product in receipt['products']:
                sample=product['sample_record_index'];image=p/'figures'/product['file']
                result=verify_png(image,bundle['paths']['local_first_cross'][sample],bundle['paths']['final_top1_cross'][sample],
                                  bundle['encoder_mask'][sample],product)
                heatmap+=result['heatmap_cells_checked'];mask_cells+=result['mask_cells_checked']
                maximum=max(maximum,result['maximum_RGB_difference_0_to_255'])
                allimages.append({'condition':n,'batch':batch,'sample':sample,'image':ident(image),**result})
        assert heatmap==38400 and mask_cells==4800
        conditions[str(n)]={'images':32,'heatmap_cells':heatmap,'mask_cells':mask_cells,'maximum_RGB_difference':maximum}
    assert len(allimages)==96
    out={'status':'A2_ALL_PNG_RAW_MATRIX_CELL_CONSISTENCY_PASS','reviewer':'/root/toys_result_audit',
         'scope':'Actual cell-center RGB lookup against raw local-first and final-Top1 matrices, plus actual encoder-mask cells. No re-render or image mutation.',
         'conditions':conditions,'total_images':96,'total_heatmap_cells':115200,'total_mask_cells':14400,'images':allimages,
         'prior_method_source':{'commit':'9ab3d0ea5d32db00e278f74c599a1943885453db',
            'path':'manifests/grid-shuffle-20261017/results-audit/verify_attention_exports.py','function':'verify_png_pixels'},
         'GPU_or_model_calls':0,'raw_evidence_writes':0,'program':ident(Path(__file__))}
    p=ROOT/'a2-png-pixel-consistency.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'images':96,'heatmap_cells':115200,'mask_cells':14400,'report':str(p)}))
if __name__=='__main__':main()
