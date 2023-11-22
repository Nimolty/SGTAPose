from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        print('self.heads', self.heads)
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            
            print('head', head)
            print('len(head_conv)', len(head_conv))
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              
              print("len(convs)", len(convs))
              
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None,repro_hm=None):
      if (pre_hm is not None) or (pre_img is not None) or (repro_hm is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm, repro_hm)
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        # print('num_stacks', self.num_stacks)
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
          # print('z', z.keys())
      return out
  
class BaseModelPlanA(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModelPlanA, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        print('self.heads', self.heads)
        for head in self.heads:
            if "wh" in head:
                continue
            classes = self.heads[head]
            head_conv = head_convs[head]
            
#            print('head', head)
#            print('len(head_conv)', len(head_conv))
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              
              print("len(convs)", len(convs))
              
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None, repro_hm=None, pre_hm_cls = None, repro_hm_cls=None):
      if (pre_hm is not None) or (pre_img is not None) or (repro_hm is not None) or (pre_hm_cls is not None) or (repro_hm_cls is not None):
        feats, pre_topk_int, repro_topk_int = self.imgpre2feats(x, pre_img, pre_hm, repro_hm, pre_hm_cls, repro_hm_cls)
      else:
        # if self.opt.phase == "ablation_wo_shared" or self.opt.phase == "abalation_shared":
        #     feats, _, _ = self.img2feats(x=x, pre_img=pre_img,pre_hm=pre_hm)
        # elif self.opt.phase == "ablation_shared_repro":
        #     feats, _, _ = self.img2feats(x=x, pre_img=pre_img,pre_hm=pre_hm,repro_hm=repro_hm)
        # else:
        #     feats = self.img2feats(x)
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        # print('num_stacks', self.num_stacks)
        for s in range(self.num_stacks):
          z = {}
          #print(self.heads)
          for head in self.heads:
              if "wh" in head:
                  continue
              z[head] = self.__getattr__(head)(feats[s])
          # z["repro_hm_topk_ind"] = repro_topk_int 
          out.append(z)
          # print('z', z.keys())
      return out


class BaseModelPlanA_Three(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModelPlanA_Three, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        print('self.heads', self.heads)
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            
#            print('head', head)
#            print('len(head_conv)', len(head_conv))
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              
              print("len(convs)", len(convs))
              
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, ppre_img=None, pre_img=None, 
                ppre_hm=None, pre_hm=None, repro_hm=None, 
                ppre_hm_cls=None, pre_hm_cls = None, repro_hm_cls=None):
      if (pre_hm is not None) or (pre_img is not None) or (repro_hm is not None) or \
         (pre_hm_cls is not None) or (repro_hm_cls is not None) or (ppre_img is not None) or (ppre_hm is not None) or (ppre_hm_cls is not None):
        feats, pre_topk_int, repro_topk_int = self.imgpre2feats(x, ppre_img, pre_img, ppre_hm, pre_hm, repro_hm, ppre_hm_cls, pre_hm_cls, repro_hm_cls)
      else:
        # if self.opt.phase == "ablation_wo_shared" or self.opt.phase == "abalation_shared":
        #     feats, _, _ = self.img2feats(x=x, pre_img=pre_img,pre_hm=pre_hm)
        # elif self.opt.phase == "ablation_shared_repro":
        #     feats, _, _ = self.img2feats(x=x, pre_img=pre_img,pre_hm=pre_hm,repro_hm=repro_hm)
        # else:
        #     feats = self.img2feats(x)
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        # print('num_stacks', self.num_stacks)
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          # z["repro_hm_topk_ind"] = repro_topk_int 
          out.append(z)
          # print('z', z.keys())
      return out
