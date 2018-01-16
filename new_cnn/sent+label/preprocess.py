#encoding:utf-8
def preprocess(src,res):
	with open(res,'w') as f_w:
		with open(src,'r') as f_r:
			for line in f_r:
				line=line.strip().decode('utf-8')
				if line:
					tmp=line.split()
					target=tmp[0]
					e1=tmp[1:3]
					e2=tmp[3:5]
					sent=tmp[5:]
					f_w.write((target+'\t'+' '.join(sent)+'\n').encode('utf-8'))

if __name__=='__main__':
	import sys
	preprocess(sys.argv[1].decode('gbk'),sys.argv[2].decode('gbk'))