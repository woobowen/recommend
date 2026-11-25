"""Write the minimal original ItemDataModule input without modifying source data."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
import argparse
from pathlib import Path
import tensorflow as tf
from runtime_common import read_jsonl, write_json, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',required=True);parser.add_argument('--data',required=True)
    args=parser.parse_args();inputs=Path(args.inputs);data=Path(args.data)
    rows=read_jsonl(inputs/'items_qwen.jsonl');out=data/'items/part-00000.tfrecord.gz'
    options=tf.io.TFRecordOptions(compression_type='GZIP')
    temporary=out.with_name(out.name+'.partial')
    with tf.io.TFRecordWriter(str(temporary),options=options) as writer:
        for row in rows:
            features={'id':tf.train.Feature(int64_list=tf.train.Int64List(value=[row['item_id']])),
                      'text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['text'].encode('utf-8')]))}
            writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
    temporary.replace(out)
    write_json(data/'item_records_manifest.json',{'items':len(rows),'path':str(out),'sha256':sha(out),
               'source_sha256':sha(inputs/'items_qwen.jsonl'),'fields':['id','text']})
    print({'item_records':len(rows),'path':str(out)},flush=True)


if __name__=='__main__':main()
