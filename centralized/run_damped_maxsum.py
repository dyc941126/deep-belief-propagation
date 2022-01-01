import os

from entities import VariableNode, FunctionNode, FactorGraph


if __name__ == '__main__':
    pth = ''
    leap = ''
    VariableNode.damp_factor = .9
    for directory in os.listdir(pth):
        cur_pth = f'{pth}/{"" if leap == "" else leap + "/"}{directory}'
        cic_pth = f'{cur_pth}/dms_cic.txt'
        bcic_pth = f'{cur_pth}/dms_bcic.txt'
        for rp in [cic_pth, bcic_pth]:
            if os.path.exists(rp):
                os.remove(rp)
        for pf in os.listdir(cur_pth):
            if not pf.endswith('.xml'):
                continue
            cic = []
            bcic = []
            fg = FactorGraph(f'{cur_pth}/{pf}', FunctionNode, VariableNode)
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