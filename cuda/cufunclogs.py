import pandas as pd
import sys
import re

class CuFuncLog(object):
    def __init__(self,header):
        self.header=header
        self.info=[]
    def addinfo(self,info):
        self.info.append(info)
    def printinfo(self):
        print(self.header)
        print(' '.join(self.info))
    def get_attributes(self):
        attr={}
        func=re.search(r'function(.+)called:',self.header)
        if func: attr['func']=func.group(1).strip()
        info=' '.join(self.info)
        gpunum=re.search(r'GPU=(\d+)',info)
        if gpunum: attr['gpu']=int(gpunum.group(1))
        time=re.search(r'Time:\s+(.+?)\s+',info)
        if time: attr['time']=time.group(1)
        items = re.findall("(\w+):(.+?)i!", info)
        for item in items:
            (name,val)=item
            attr[name]=val
        return attr

class CuFuncParser(object):
    def __init__(self,logpath):
        self.logpath=logpath
        self.cufuncs=[]
        self.parse()
    def parse(self):
        with open(self.logpath) as logf:
            lines=logf.readlines()
        cuobj=None
        for line in lines:
            line=line.strip()
            if re.match(r'I!',line):
                if cuobj is not None:
                    self.cufuncs.append(cuobj)
                cuobj=CuFuncLog(line)
            elif re.match(r'i!',line):
                cuobj.addinfo(line)
    def printall(self):
        for cufunc in self.cufuncs:
            cufunc.printinfo()
            print(cufunc.get_attributes())
    def printgpu(self):
        for cufunc in self.cufuncs:
            #cufunc.printinfo()
            attr=cufunc.get_attributes()
            #if 'gpu' in attr.keys():
            if True:
                print('\n',attr['func'])
                if 'dimA' in attr.keys():
                    print(attr['dimA'])
                print(attr)

if __name__ == '__main__':
    cuparser=CuFuncParser(sys.argv[1])
    cuparser.printgpu()
