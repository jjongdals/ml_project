# **머신러닝 프로젝트 구현**


📝소학회, 스터디 코드 구현 파일 기록 정리

## ‼️ ***실습 코드 보기에 앞서***





## pytorch 설치

GPU 연산을 사용할 예정이라 ***pytorch 설치***

```bash
python3 -m pip install torch
```

## 적용

```python
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
```

## MPS 지원확인
```python
print (f"PyTorch version:{torch.__version__}") # 버전확인
print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}") 
print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}") 
!python -c 'import platform;print(platform.platform())'
```
`2,3번째의 print문이 true가 나오면 지원확인 가능`


## License
[MIT](https://choosealicense.com/licenses/mit/)
