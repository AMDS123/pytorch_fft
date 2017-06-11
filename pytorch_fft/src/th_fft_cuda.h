int th_Float_fft(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2, int d);
int th_Float_ifft(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2, int d);
int th_Double_fft(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2, int d);
int th_Double_ifft(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2, int d);

int th_Float_fft2(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2, int d);
int th_Float_ifft2(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2, int d);
int th_Double_fft2(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2, int d);
int th_Double_ifft2(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2, int d);