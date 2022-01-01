import os

from advp_entities import ADVPFunctionNode, ADVPVariableNode, ADVPFactorGraph


if __name__ == '__main__':
    pth = ''
    leap = ''
    for directory in os.listdir(pth):
        cur_pth = f'{pth}/{"" if leap == "" else leap + "/"}{directory}'
        cic_pth = f'{cur_pth}/advp_cic.txt'
        bcic_pth = f'{cur_pth}/advp_bcic.txt'
        for rp in [cic_pth, bcic_pth]:
            if os.path.exists(rp):
                os.remove(rp)
        for pf in os.listdir(cur_pth):
            if not pf.endswith('.xml'):
                continue
            cic = []
            bcic = []
            fg = ADVPFactorGraph(f'{cur_pth}/{pf}', ADVPFunctionNode, ADVPVariableNode)
            for it in range(1000):
                cost = fg.step()
                cic.append(int(cost))
                if len(bcic) == 0:
                    bcic.append(cic[-1])
                else:
                    bcic.append(min(bcic[-1], cic[-1]))
            with open(cic_pth, 'a') as rf:
                rf.write(str(cic) + '\n')
            with open(bcic_pth, 'a') as rf:
                rf.write(str(bcic) + '\n')