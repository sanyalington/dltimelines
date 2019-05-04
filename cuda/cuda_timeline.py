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
    def get_copy_activity(self):
        gpudf=self.get_gpu_activity()
        copydf=gpudf[gpudf.Size.notnull()]
        #filter(["Size","Throughput","SrcMemType","DstMemType","Name"])
        print(copydf.Name.unique())
        print(copydf.Size.unique())
        largetrf=copydf[copydf.Size>1]
        largetrf=largetrf[largetrf.Name=='[CUDA memcpy HtoD]']
        print(len(largetrf))
        largetrf.to_csv('/tmp/largetrf.csv',index=None)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tracefile")
    args = parser.parse_args()
    tracer=CudaTracer(args.tracefile)
    tracer.get_copy_activity()

