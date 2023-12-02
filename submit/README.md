1. 모델 학습 코드 & validation 코드     
- mission1.ipynb    
- mission2.ipynb      
- mission3.ipynb

2. 모델 가중치 저장 파일(pt_structure.txt 참고) 
- mission1.pt      
> 이 파일에는 resnet18에 해당하는 가중치만 존재하기 때문에 모델을 불러오시려면 model.load_state_dict(torch.load(PATH))를 쓰시면 됩니다.     
- mission2.pt     
> 이 파일에는 resnet18, resnet34, resnet50, resnet101, resnet152의 가중치가 모두 포함되어있기 때문에 모델을 불러오시려면 model.load_state_dict(torch.load(PATH)[\'resnet18\']) 와 같이 사용하시면 됩니다.      
- mission3.pt      
> 이 파일에는 mission2.pt와 같은 구조로 되어있습니다.

3. model.py       
mission2의 최종모델에 ensemble이 사용되기 때문에 그 구조를 만들어 놨습니다. mission3의 경우에는 resnet152 단일 모델만 최종모델로 사용했기 때문에 만들어놓은 코드대로 하지 않으셔도 됩니다.

4. pt_structure.txt     
.pt형식의 파일들에 어떤식으로 가중치가 저장되어있는지 써놨습니다.

5. normalize_stat.txt
mission1, 2, 3 에 사용된 mean값과 std값이 들어있습니다.

6. takeout발표자료.pptx
발표자료 입니다.

