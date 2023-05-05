import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# CSV 파일 읽기
data = pd.read_csv('debug_autolog.csv', header=None)

# 두 번째, 세 번째, 네 번째 열 선택
columns = data.iloc[:, 1:4]

# 열 이름 변경
columns = columns.rename(columns={1: 'prev_angle', 2: 'model_output*k', 3: 'diff_angle'})

# 3개의 서브플롯 생성
fig, axs = plt.subplots(3)

# 각 서브플롯에 플롯 생성 및 이름 설정
axs[0].plot(columns['prev_angle'])
axs[0].set_title('prev_angle')
axs[1].plot(columns['model_output*k'])
axs[1].set_title('model_output*k')
axs[2].plot(columns['diff_angle'])
axs[2].set_title('diff_angle')

# x축과 y축의 major_locator와 minor_locator 초기화
for ax in axs:
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())

# 격자 간격 조절
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
axs[2].xaxis.set_major_locator(ticker.MultipleLocator(5))

# 격자 색상과 투명도 조절
for ax in axs:
    ax.grid(True, which='major', color='gray', alpha=0.5)

# 플롯 보여주기
plt.show()