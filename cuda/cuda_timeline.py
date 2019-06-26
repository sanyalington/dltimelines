import pandas as pd
import argparse
import numpy as np

class CudaTracer():
    def __init__(self,tracepath):
        self.tracepath=tracepath
        self.tracedf=self.import_trace()
    def import_trace(self):
        colnames=["Start","Duration","GridX","GridY","GridZ","BlockX","BlockY","BlockZ","RegistersPerThread","StaticSMem","DynamicSMem","Size","Throughput","SrcMemType","DstMemType","Device","Context","Stream","Name","Correlation_ID"]
        df = pd.read_csv(self.tracepath,skiprows=5,header=None,names=colnames)
        #first 3 rows typically contain prof cmd line
        #next 2 have header info
        values = {'Device': 'api'}
        df.fillna(value=values,inplace=True)
        print(df.tail(1))
        #print(df.dtypes)
        return df
    def get_gpu_activity(self):
        gpus=list(self.tracedf.Device.unique())
        gpus.remove('api')
        gpus=sorted(gpus)
        print('Devices found:',gpus)
        gpu0=gpus[0]
        print('Default device:',gpu0)
        gpudf=self.tracedf.copy()
        gpudf=gpudf[gpudf.Device==gpu0]
        print(gpudf.head())
        print(gpudf.tail())
        print('GPU trace length',len(gpudf))
        print('Contexts',gpudf.Context.unique())
        print('Contexts',gpudf.Stream.unique())
        return gpudf
    def get_large_h2d_copy_activity(self,min_trf_size_mb=1):
        gpudf=self.get_gpu_activity()
        copydf=gpudf[gpudf.Size.notnull()]
        #filter(["Size","Throughput","SrcMemType","DstMemType","Name"])
        print(copydf.Name.unique())
        print(copydf.Size.unique())
        largetrf=copydf[copydf.Size>min_trf_size_mb]
        largetrf=largetrf[largetrf.Name=='[CUDA memcpy HtoD]']
        print(len(largetrf))
        largetrf.to_csv('/tmp/largetrf.csv',index=None)
        return largetrf

def resnet50_analysis(tracer):
    batches=tracer.get_large_h2d_copy_activity()
    batches['batch_time']=batches.Start.shift(-1)-batches.Start
    print('Num batches found:', len(batches))
    print(batches)
    analysis_batch_num=10
    batch_start_time=batches['Start'].iloc[analysis_batch_num]
    nxt_batch_start_time=batches['Start'].iloc[analysis_batch_num+1]
    
    print(batch_start_time,nxt_batch_start_time)
    batchdf=tracer.get_gpu_activity()
    batchdf=batchdf[batchdf.Start>=batch_start_time]
    batchdf=batchdf[batchdf.Start<=nxt_batch_start_time]
    batchdf.to_csv('/tmp/batchdf.csv',index=None)
    print(batchdf.tail())
    print(batchdf.Stream.value_counts())
    batchdf_main=batchdf[batchdf.Duration>20]
    batchdf_main.to_csv('/tmp/batchdf_main.csv',index=None)
    print(batchdf.Duration.sum())
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tracefile")
    args = parser.parse_args()
    tracer=CudaTracer(args.tracefile)
    resnet50_analysis(tracer)

