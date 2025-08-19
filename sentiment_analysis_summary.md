# 深圳地铁服务维度情感分析结果

## 概述
根据 `merged_weibo_content.txt` 中的微博内容，我们对深圳地铁的6个服务维度进行了情感分析，生成了12个CSV文件。

## 生成的CSV文件

### 可靠性 (Reliability)
- `reliability_positive_sentiment_scores.csv` - 可靠性正面情感得分
- `reliability_negative_sentiment_scores.csv` - 可靠性负面情感得分

### 拥挤度 (Crowdedness)  
- `crowdedness_positive_sentiment_scores.csv` - 拥挤度正面情感得分
- `crowdedness_negative_sentiment_scores.csv` - 拥挤度负面情感得分

### 员工服务 (Staff)
- `staff_positive_sentiment_scores.csv` - 员工服务正面情感得分
- `staff_negative_sentiment_scores.csv` - 员工服务负面情感得分

### 舒适度 (Comfort)
- `comfort_positive_sentiment_scores.csv` - 舒适度正面情感得分
- `comfort_negative_sentiment_scores.csv` - 舒适度负面情感得分

### 安全性 (Safety)
- `safety_positive_sentiment_scores.csv` - 安全性正面情感得分
- `safety_negative_sentiment_scores.csv` - 安全性负面情感得分

### 服务设施 (Service Facility)
- `service_facility_positive_sentiment_scores.csv` - 服务设施正面情感得分
- `service_facility_negative_sentiment_scores.csv` - 服务设施负面情感得分

## CSV文件格式
每个CSV文件都包含两列：
- `name`: 地铁站名称
- `[positive/negative]_sentiment_score`: 对应的情感得分

如果某个站点在该维度没有相关的情感提及，得分列将为空。

## 分析统计
- 总共处理了 58,758 行微博数据
- 共有 371 个地铁站点被提及
- 分析了以下关键词：

### 可靠性 (Reliability)
- 正面：准时、可靠、准点、正常运行、稳定、按时、不延误
- 负面：延误、晚点、故障、停运、不准时、取消、中断、停止、断电、维修

### 拥挤度 (Crowdedness)
- 正面：空旷、不拥挤、有座位、人少、宽敞、舒适、空座、排队有序
- 负面：拥挤、挤、人多、爆满、人山人海、站不稳、挤不上、排队、满员、没座位

### 员工服务 (Staff)
- 正面：服务好、态度好、热情、礼貌、帮助、友善、专业、微笑、耐心、周到
- 负面：态度差、服务差、不礼貌、冷漠、不耐烦、粗暴、不专业、不理人、无礼

### 舒适度 (Comfort)
- 正面：舒适、凉爽、温度适宜、座位舒服、安静、环境好、干净、明亮
- 负面：不舒服、太冷、太热、闷热、吵闹、脏、座位硬、环境差、空调太冷、冷死、热死

### 安全性 (Safety)
- 正面：安全、放心、保障、防护、稳定、安心
- 负面：危险、不安全、事故、摔倒、受伤、撞击、夹到、滑倒、碰撞

### 服务设施 (Service Facility)
- 正面：设施好、设备新、电梯正常、厕所干净、wifi好、充电方便、无障碍、指示清楚
- 负面：设施差、设备坏、电梯故障、没厕所、无wifi、不能充电、无障碍差、指示不清

## 前10个最常被提及的站点
1. 深圳: 33,510 次提及
2. 深圳北站: 1,060 次提及
3. 深圳北: 1,060 次提及
4. 福田: 893 次提及
5. 宝安: 709 次提及
6. 机场: 590 次提及
7. 坪洲: 564 次提及
8. 固戍: 537 次提及
9. 西乡: 490 次提及
10. 罗湖: 443 次提及

## 注意事项
- 所有CSV文件的格式与 `station_sentiment_rank.csv` 保持一致
- 包含了原始文件中的所有371个地铁站点
- 情感得分基于关键词匹配和频次计算
- 得分为0的情况下，CSV中该字段为空
