# coding=utf-8
import os
import json
import time
from tqdm import tqdm
import unicodedata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import traceback


class FormatHandler():
    def __init__(self, input_path, output_path, dataset_name):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name

    def get_file_list(self) -> list:
        """或取输入路径下的全部文件，可以放一些文件名判断逻辑"""
        files = os.listdir(self.input_path)
        files = [i for i in files if ".json" in i]
        return files
    
    def process_one_line(self, line, fout) -> bool:
        """处理一行数据，子类必须实现"""
        raise NotImplementedError
    
    def process_one_file(self, file_path) -> (int, int):
        """处理一份文件"""
        line_count = 0
        jump_count = 0
        with open(self.output_path, "a") as fout:
            with open(self.input_path + "/" + file_path, "r", encoding="utf-8") as fin:
                # for line in tqdm(fin):
                for line in fin:
                    line_count += 1
                    self.process_one_line(line, fout)
        return line_count, jump_count
    
    def process_all(self):
        """处理全部文件"""
        st = time.time()
        line_count_all = 0
        jump_count_all = 0
        file_list = self.get_file_list()
        print("[log][{}] number of files is {:d}".format(self.dataset_name, len(file_list)))
        # for file in tqdm(file_list):
        for file in file_list:
            line_count, jump_count = 0, 0
            try:
                line_count, jump_count = self.process_one_file(file)
            except Exception as e:
                print("[exception][{}] process file {} failed: {}".format(self.dataset_name, file, e))
                print(traceback.format_exc())
            line_count_all += line_count
            jump_count_all += jump_count
        print("[log][{}] timecost is {:.2f} s!".format(self.dataset_name, time.time() - st))
        print("[log][{}] line_count is {:d}, jump_count is {:d}".format(self.dataset_name, line_count_all, jump_count_all))

    def quality_assurance(self, line) -> bool:
        """确保一段文字的基本质量, todo"""
        # 1. 字数过少
        if len(line) < 20:
            return False

        # 2. 特殊符号使用过多
        return True

    def zh_process(self, line) -> str:
        """初步的中文文本处理, todo"""
        # 0. None 处理成空字符串
        if line is None:
            return ""
        # 1. 半角全角统一。

        # 2. 英式句号、逗号转换为中式。可能并不必要。

        # 3. unicode 统一
        line = unicodedata.normalize("NFKC", line)
        # 4. 去除\r（针对 baidu_QA）
        line = line.replace("\r", "")
        # 5. 替换\n
        line = line.replace("\n\n", "\n")
        return line


class BaiduBaikeFormatHandler(FormatHandler):
    """example data
    {"title": "红色食品", "summary": "红色食品是指食品为红色、橙红色或棕红色的食品。科学家认为，多吃些红色食品可预防感冒。红色食品有红柿椒、西红柿、胡萝卜、红心白薯、红果（山楂）、红苹果、草莓、红枣、老南瓜、红米、柿子等。 有治疗缺铁性贫血和缓解疲劳的作用，对乳腺癌等肿瘤疾病有防治作用，给人以兴奋感，有增加食欲，光洁皮肤，增强表皮细胞再生和防止皮肤衰老，预防感冒等作用。", "sections": [{"title": "简介", "content": "红色食品富含番茄红素、胡萝卜素、铁和部分氨基酸，是优质蛋白质、碳水化合物、膳食纤维、B族维生素和多种无机盐的重要来源，可以弥补粳米、白面中的营养缺失。经常食用红色食品，可以进一步提高对主食中营养的利用率，山植等食品还有治疗癌症的功效。被称为“红色生力军。”营养学家认为，红色蔬果最典型的优势在于它们都是富含天然铁质的食物，例如我们常吃的樱桃、大枣等都是贫血患者的天然良药，也适合女性经期失血后的滋补。所以，红色蔬果，女人尽可放心多吃。红色食品中还含有致病微生物的“杀手”——巨噬细胞，可以有效地抵御感冒病毒等微生物，增强人体抵抗感冒的能力。  红色食品\n在所有的果蔬当中，名声最好的莫过于苹果。西方有“One apple a day，keeps the doctors away．”的说法，因为苹果性情温和，含有各种维生素和微量元素，是所有的水果中最接近完美的一个。\n还有一种说法：红色食品是相对于绿色食品而言的，指对人体有害的食品，如各种有毒有害、腐败变质、添加非食用物质的食品。红色食品危害人体健康乃至生命安全，对人体健康亮起了红灯，应当大力查处。"}, {"title": "作用", "content": "这些食品中富含β-胡萝卜素和维生素A，对孩子上皮组织和呼吸道粘膜有很强的保护作用，可提高预防感冒的能力。\n假如你生来体质较弱，易受感冒病毒的困扰，或者已经被感冒缠上了，红色食品会助你一臂之力，天生具有促进人体健康卫士之一的巨噬细胞活力的功能，巨噬细胞乃是感冒病毒等致病微生物的“杀手”，其活力增强了，感冒病毒自然难以在人体内立足，更谈不上生长繁殖了。至于颜色较辣椒稍浅一些的胡萝卜，所含的胡萝卜素可在体内转化为维生素A，发挥护卫人体上皮组织如呼吸道黏膜的作用，常食之同样可以增强人体抗御感冒的能力。除了红辣椒、胡萝卜外，苋菜、洋葱、红枣、番茄、红薯、山楂、苹果、草莓、老南瓜、红米等亦具此功。"}, {"title": "红色食品与感冒", "content": "冬令时节，气候寒冷，万物收藏，人的机体生理功能处于降低、抑制、收缩状态，易患感冒，吃红色食品可扶正祛邪，增强免疫力，预防得病。[1]\n蔬菜中的红萝卜、红辣椒、番茄、红薯、红枣、红苋菜等红色食品中，富含β-胡萝卜素，不但能清除人体氧自由基，而且参与合成维生素A，对人体上皮组织和 呼吸道黏膜有很强的保护作用。推荐以下几个食疗方———\n★番茄猪肝汤\n用料：猪肝250克，虾仁25克，蘑菇40克，鸡蛋1只，番茄150克，黄酒、葱段、姜片、胡椒粉、精盐适量。\n制法：将猪肝切去筋膜洗净，切丁后加上酒、姜汁、蛋液、盐、胡椒粉，搅打成浆。用旺火蒸10—15分钟至结膏。清水加虾仁、黄酒沸煮5分钟后倒入蘑菇、番茄丁和肝膏，再煮沸，调味即可。\n功用：养肝明目，增强免疫力。用于防感冒、防治夜盲症及免疫力低下者，以及甲亢。\n方解：猪肝有养肝明目、增强免疫力；蘑菇补益脾胃，益阴养肝，降压，降脂，润燥化痰，增加白细胞；虾仁、番茄均有增强免疫力的食品，番茄可增强营养，减少感冒的发生。\n★红萝卜炖牛肉\n用料：牛肉250克、红萝卜250克。\n制法：牛肉切成小块，加黄酒、姜、葱等配料，再加入红萝卜块，炖熟，即可食用。\n功用：益气养胃、强健筋骨、增强免疫力。适用于防感冒及免疫力低、虚损消瘦、腰膝酸软者。\n方解：牛肉补脾胃、益气血、强筋骨，红萝卜增强免疫力，防感冒，健脾，补血，助消化。\n★蜂蜜红萝卜汁\n用红萝卜汁与蜂蜜各半制成混合汁剂，每天饮用3次，每次1汤匙。可防治伤风、感冒和咽喉炎。胡萝卜能提供丰富的维生素A，具有促进机体正常生长与繁殖、维护上皮组织、防止感冒及保持视力正常。蜂蜜能补中，润燥，止痛，解毒，清热。"}, {"title": "红色食品与红肉", "content": "红色食品是指外表呈红色的果蔬和“红肉”类。红色果蔬包括红辣椒、西红柿、红枣、山楂、草莓、苹果等，红色果蔬含有糖和多种维生素，尤其富含维生素C。“红肉”指牛肉、猪肉、羊肉及其制品。\n红色果蔬中的辣椒具有温中散寒，开胃除湿之功效，辣椒中的辣椒素能刺激唾液和胃肠道消化液的分泌，还能刺激心血管系统，使心跳加快，血液循环加速，因此在寒冷环境有祛风除湿的作用。风寒型感冒病人食用辣椒汤能帮助发汗，有利于感冒的康复，但是胃肠疾病、结核病人则不适合食用。西红柿-在国外享有“金苹果”之称，具有较高的价值。由于西红柿含有94%左右的水分，生吃能防治中暑，止渴生津，凉血解毒的作用，但食西红柿时尽量少放盐，为了避免维生素的破坏，做汤时最好等水开了再下西红柿，而且忌食未成熟的西红柿胃，胃肠虚寒者即慢性腹泻和消化不良者应忌食之。红枣在国外被称为“天然维生素丸”，具有很好的补血功效，能安神和补益脾胃，但胃肠积滞和患有牙齿疾病者应忌食。食用红枣时不宜与鱼同食，同食易引起腹部胀痛。\n红色食品中的肉类即所谓“红肉”，主要含蛋白质和脂肪及其它无机盐等，因此具有丰富的营养价值。不过“红肉”致癌，世界癌症研究基金会建议食用“红肉”时，每日每人撮入量应少于80克，这是因为“红肉”在烧烤、烙制、煎炸时，其表面产生多种杂环胺——致癌物。"}, {"title": "好处", "content": "红色不但能让人联想到爱情和激情，还是一种与心脏、大脑和泌尿系统的健康有关的颜色。红色的水果和蔬菜对我们的身体健康大有裨益。 红色的果蔬主要含有丰富的植物化学成分，包括抗细胞因子、抗氧化剂和番茄红素(一种产生红色的色素)。这些成分能够预防癌症，特别是肺癌、前列腺癌和消化道癌。它们还可以延缓衰老，并且有利于防止黄斑变性。黄斑变性是导致65岁以上老年人失明的主要诱因。\n1、草莓含有80%的水分、丰富的维生素C、少量膳食纤维和钾。由于含糖量很低，因此经常出现在减肥食谱中。大量抗坏血酸、凝集素和果胶使它成为降低血液中胆固醇含量的理想食物。传统医学中草莓可以作为润肤剂、净化剂和利胆剂，还具有镇咳和抗风湿的作用。人们还认为它有抗贫血和滋补的功效，可以促进机体生长。草莓叶子的浸剂还可以用于肠道消炎。\n2、每天生食6至1 2颗樱桃对痛风和尿酸过多有显著疗效。樱桃的果汁有利于治疗腹泻和结肠炎，所含的抗氧化剂具有保健作用。樱桃含有的主要养分和膳食纤维较少，但是维生素B的含量不低。在矿物质中，钾和铁的含量较高。\n3、西瓜是水分含量最高的水果，高达其总重量的95%，因此可以促进肾脏更好地发挥功能，将废物和有毒物质排出体外。\n4、覆盆子含有丰富的维生素E、多种植物营养素和不可溶性纤维。除了具有利尿和通便作用，它还可以用于治疗风湿。\n5、红苹果富含果胶、糖分和维生素C。此外，由于它是温和的通便剂，所以还具有特殊的医疗效用，可以用于治疗肠道功能紊乱。因此，自然医学认为红苹果可以抗腹泻、贫血和哮喘。它还能够缓解神经系统紧张，促进睡眠。每天晚上吃一个红苹果有助于迅速入睡。它还对希望保持体形的人有用，因为几乎不含脂肪，每100克只有不到58卡路里的热量。据法国图卢兹大学的研究结果显示，每天吃一个大苹果可以在8个星期内使胆固醇水平降低。\n6、红辣椒中抗氧化剂、硒和维生素C的含量很高，甚至高于柑桔和柠檬等酸味水果。红辣椒所含的膳食纤维能够控制血液中的胆固醇和葡萄糖，还可以改善肠道功能。\n7、红萝卜有益于治疗呼吸系统疾病，例如咽炎和喉炎，还可以减轻喉咙嘶哑。在柠檬的辅助下，它还可以用来防治哮喘和鼻窦炎。萝卜酒具有清除肾结石和治疗肝脏和胆囊疾患的作用。红萝卜含有钾和少量铁，不含脂肪，每100克只含有15卡路里的热量。它非常适于制作凉拌沙拉，配上柠檬和盐就是一道佳肴。此外，吃红萝卜还可以控制前列腺癌变。\n8、番茄含有番茄红素和大量抗氧化剂，能够降低患上慢性疾病的危险，尤其是前列腺癌和心血管疾病。番茄具有提神、助消化和抗炎的作用。用它制作的沙拉、酱汁和菜泥可以帮助患有胃炎和胃溃疡的人更好地消化不易消化吸收的食物。番茄的热量很低，含有维生素C。它富含的番茄红素可以防止罹患前列腺癌。如果使用食用油烹调番茄，还可以增强这种功效。\n"}], "tags": ["饮食", "食品", "食疗", "科学", "健康", "食品类型"], "url": "http://baike.baidu.com/view/0010.htm"}
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(BaiduBaikeFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def process_one_line(self, line, fout) -> bool:
        text = ""
        data = json.loads(line)
        title = self.zh_process(data["title"])
        content = self.zh_process(data["summary"])
        if title == "" or content == "":
            return False
        text = title + "：" + content
        for section in data["sections"]:
            text += self.zh_process(section["title"]) + "：" + self.zh_process(section["content"])
        if not self.quality_assurance(text):
            return False
        d = {"text": text}
        fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return True

class BaiduQAFormatHandler(FormatHandler):
    """example data
    {"qid": "qid_5982723620932473219", "category": "教育/科学-理工学科-地球科学", "title": "人站在地球上为什么没有头朝下的感觉 ", "desc": "", "answer": "地球上重力作用一直是指向球心的，因此\r\n只要头远离球心，人们就回感到头朝上。"}
    {"qid": "qid_5679706523376347837", "category": "娱乐-宠物", "title": "我的小baby", "desc": "我的小baby-辛巴。温顺可爱的，两个月大的小家伙，第一次养狗，该注意什么呢？求指教～[爱你]", "answer": "勤洗澡，养成好的卫生习惯"}
    特点是用户提出的问题或者回答，标点符、语法使用不够规范，并且会混入表情包，比如[爱你]。
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(BaiduQAFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def process_one_line(self, line, fout) -> bool:
        data = json.loads(line)
        title = self.zh_process(data["title"])
        answer = self.zh_process(data["answer"])
        # text = title + "：" + answer  # todo, 其实title末尾可能已经有标点符号了。
        text = "问题：\n{}\n回答：\n{}".format(title, answer)
        
        if not self.quality_assurance(text): # todo 质量筛选
            return False
        d = {"text": text}
        fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return True

class BELLEFormatHandler(FormatHandler):
    """example data
    {"instruction": "编写一个Python函数，计算给定列表的平均值。\nnums = [3, 5, 2, 8, 10]", "input": "", "output": "可以根据以下步骤编写一个Python函数来计算给定列表的平均值：\n1. 定义函数，它将取一个列表作为参数。\n2. 计算列表中所有元素的总和。\n3. 计算列表中元素的数量。\n4. 将总和除以元素数量来计算平均值。\n5. 返回平均值作为函数输出。\n以下是Python代码实现：\n```python\ndef calculate_mean(nums):\n  total = sum(nums)\n  count = len(nums)\n  mean = total / count\n  return mean\nnums = [3, 5, 2, 8, 10]\nprint(calculate_mean(nums))\n```\n在上面的代码片段中，我们首先定义了一个叫做`calculate_mean`的函数，它将一个列表作为输入参数。然后，我们使用了Python内置的`sum`函数来计算列表中所有元素的总和，使用`len`函数计算列表中元素的数量，然后将总和除以元素数量来计算平均值。最后，我们使用`return`语句将平均值作为函数输出。\n代码中的下一行创建一个名为`nums`的变量，其中包含我们要计算平均值的数字列表。最后，我们使用`print`函数输出函数的结果，即给定列表的平均值，结果为`5.6`。"}
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(BELLEFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, line, fout) -> bool:
        data = json.loads(line)
        instruction = self.zh_process(data["instruction"])
        output = self.zh_process(data["output"])
        text = instruction + "\n" + output # todo: 更好的连接词？
        if not self.quality_assurance(text):
            return False
        d = {"text": text}
        fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return True

class BELLEConversationsFormatHandler(FormatHandler):
    """example data
    {"conversations": [{"from": "human", "value": "针对健身房的新手，设计一套适合
他们的健身器械使用指南，包括安全应用、正确姿势等方面。"}, {"from": "assistant", "value": "健身器械使用指南\n1. 开始前，请先进行热身运动。这会帮助你的身体>适应运动，并减少受伤的风险。\n2. 在使用健身器械前，确保你已经了解了其使用方法
。请阅读说明书或咨询教练以获得正确的使用技巧。\n3. 谨防过度使用或过度挑战你的
身体。 如果你觉得有些动作太难或太重，请添加锻炼计划，以逐步提高动作难度。\n4. 使用合适的装备。 确保你拥有合适的运动鞋和舒适的运动服。 不要在裸露的脚或短裤
上进行重量训练。\n5. 在健身器械上使用安全装置。 这些通常用于保护你的身体免受>不当操作造成的损伤。 例如，重量训练中，你需要使用杠铃和负重时，一定要使用卡子
来防止重量滑落。\n6. 注意正确的姿势。 如果你的姿势是错误的，那么你的身体很容>易被伤害到，你也可能无法获得最佳的锻炼效果。 至关重要的是，保持直立的身体，保
持头部和颈部的稳定，并使用合适的重量。\n7. 保持合理的呼吸方式。 无论何时进行>训练，都必须保持正常呼吸。 当你需要用力时，呼气； 当你放松时，吸气。\n8. 安全
存放器械。 在使用健身器械后，你需要把它们归还给适当的位置，以便其他人可以使用
它们。\n总之，健身器械的正确使用是关键之一，如果不健康和不安全，它们将无法帮>助您达到您所需的健康成果。 选择适当的训练计划，并为训练提供足够的时间，以备逐
渐适应新方法。 对于任何问题，请向教练咨询。"}], "id": "66182880"}
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(BELLEConversationsFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, line, fout) -> bool:
        text_list = []
        text_list.append("以下是一些对话：")
        data = json.loads(line)
        conversations = data["conversations"]
        temp_pair = ""
        for index, item in enumerate(conversations):
            if index % 2 == 0 and item["from"] == "human":
                temp_pair += "问题：\n{}\n".format(self.zh_process(item["value"]))
            elif index % 2 == 1 and item["from"] == "assistant":
                temp_pair += "回答：\n{}".format(self.zh_process(item["value"]))
                text_list.append(temp_pair)
                temp_pair = ""
        text = "\n".join(text_list)
        if not self.quality_assurance(text):
            return False
        d = {"text": text}
        fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return True
    
class FireflyFormatHandler(FormatHandler):
    """example data
    {"kind": "NLI", "input": "自然语言推理：\n前提：他觉得自己已离开了很久的时间,而这里的人和事竟然依旧,似乎是在等着他归队,真叫人倍感温馨\n假设：他是个酒吧老板。\n两者是什么关系？", "target": "中立"}
    {"kind": "NER", "input": "下文中存在哪些实体：\n如今两国又进行耗资巨大的核军备竞赛，这势必严重影响本国经济发展和人民生活改善。", "target": "不存在实体"}
    {"kind": "Summary", "input": "在上海的苹果代工厂，较低的基本工资让工人们形成了“软强制”的加班默契。加班能多拿两三千，“自愿”加班成为常态。律师提示，加班后虽能获得一时不错的报酬，但过重的工作负荷会透支身体，可能对今后劳动权利造成不利影响。\n输出摘要：", "target": "苹果代工厂员工调查：为何争着“自愿”加班"}
    {"kind": "Cot", "input": "请一步一步认真地思考下面的问题：\nFran is in charge of counting votes for the book club's next book, but she always cheats so her favorite gets picked. Originally, there were 10 votes for Game of Thrones, 12 votes for Twilight, and 20 votes for The Art of the Deal. Fran throws away 80% of the votes for The Art of the Deal and half the votes for Twilight. What percentage of the altered votes were for Game of Thrones?", "target": "First find the total number of The Art of the Deal votes Fran throws away: 80% * 20 votes = 16 votes. Then subtract these votes from the total number of The Art of the Deal votes to find the altered number: 20 votes - 16 votes = 4 votes. Then divide the total number Twilight votes by 2 to find the altered number of votes: 12 votes / 2 = 6 votes. Then add the altered number of votes for each book to find the total altered number of votes: 6 votes + 4 votes + 10 votes = 20 votes. Then divide the number of votes for Game of Thrones by the total altered number of votes and multiply by 100% to express the answer as a percentage: 10 votes / 20 votes * 100% = 50%.\n因此，最终的答案是：\n50"}
    {"kind": "Couplet", "input": "根据上联给出下联：风阅大江头，风流何处？江流何处", "target": "下联：人浮沧海外，人在天涯，海在天涯"}
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(FireflyFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, line, fout) -> bool:
        data = json.loads(line)
        kind = self.zh_process(data["kind"])
        input_str = self.zh_process(data["input"])
        target_str = self.zh_process(data["target"])
        text = "\n".join(["任务类型：",kind, "问题：", input_str, "回答：", target_str])  # todo，更好的连接词？
        if not self.quality_assurance(text):
            return False
        d = {"text": text}
        fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return True

class MossFormatHandler(FormatHandler):
    """example data
    {"conversation_id": 1, "category": "Brainstorming", "conversation": [{"human": "如何保障工作中遵循正确的安全准则？", "assistant": "为了保障工作中遵循正确的安全准则，以下是一些建议：\n\n1.了解相关安全规定和标准：了解相关的安全规定和标准，并遵守它们。这可以包括公司和政府的安全标准，以及行业标准和最佳实践。\n\n2.培训和教育：确保您和您的同事接受了必要的培训和教育，以了解正确的安全准则和行为。\n\n3.使用正确的工具和设备：确保您使用正确的工具和设备，并且它们得到了正确的维护和保养。\n\n4.个人防护装备：确保您和您的同事穿戴正确的个人防护装备，如安全鞋、透明眼镜或面罩、手套等。\n\n5.创建一个安全文化：鼓励个人对安全的责任感和意识，并创建一个安全文化，使人们始终关注他们自己和他人的安全。\n\n6.持续监测和改进：持续监测和改进安全准则和程序，以确保它们保持最新，并适应变化的情况。\n\n这些措施可以帮助您和您的同事在工作中保障正确的安全准则，确保您的工作场所是一个安全的环境。"}, {"human": "如何应对突发紧急情况？", "assistant": "突发紧急情况可能会在任何时间发生，为了应对这些情况，以下是一些建议：\n\n1.保持冷静：保持冷静和清醒，以便能够更好地应对情况。\n\n2.评估情况：评估紧急情况的性质和范围，以便了解应该采取哪些行动。\n\n3.触发应急计划：如果您的机构或公司有应急计划，立即触发它，并按照计划的步骤行动。\n\n4.联系相关人员：通知必要的人员，例如紧急服务、管理层或其他相关人员，并与他们保持联系。\n\n5.遵循安全准则：在处理紧急情况时，请遵循正确的安全准则，确保自己和其他人的安全。\n\n6.提供及时信息：尽快向相关人员提供准确的信息，以便他们能够做出正确的决策。\n\n7.合作应对：与其他人合作应对紧急情况并相互支持。\n\n总之，应对突发紧急情况需要保持冷静和清醒，评估情况并采取适当的行动。同时，需要遵循正确的安全准则，并与其他人合作应对紧急情况。"}, {"human": "在网络安全方面，如何避免遭受黑客攻击？", "assistant": "以下是一些避免遭受黑客攻击的建议：\n\n1.更新安全软件：确保您的操作系统、应用程序和安全软件是最新的，并随时更新它们。这可以帮助您及时发现和修复安全漏洞。\n\n2.使用强密码：使用强密码，包括字母、数字和特殊字符。不要使用容易猜测的密码，如生日、姓名或常用单词。\n\n3.使用多因素身份验证：使用多因素身份验证来增加账户的安全性，例如发送短信验证码或使用指纹识别。\n\n4.小心附件和链接：不要打开未知的附件或链接，因为它们可能包含恶意软件。\n\n5.备份重要文件：定期备份重要文件，并将其存储在安全的位置。\n\n6.保护个人信息：不要在互联网上公开个人信息，例如出生日期、地址或社交安全号码。\n\n7.使用加密连接：在访问敏感信息时，使用加密连接，如HTTPS。\n\n8.教育员工：向员工提供网络安全培训，并确保他们了解正确的安全措施和行为。\n\n总之，避免遭受黑客攻击需要采取多种措施，包括更新安全软件、使用强密码、使用多因素身份验证、小心附件和链接、备份重要文件、保护个人信息、使用加密连接以及教育员工。这些措施可以帮助您保护个人信息和数据免受黑客攻击。"}]}
    基于MOSS-002内测阶段采集的约10万用户输入数据和gpt-3.5-turbo构造而成，相比moss-002-sft-data，moss-003-sft-data更加符合真实用户意图分布，包含更细粒度的有用性类别标记、更广泛的无害性数据和更长对话轮数，约含110万条对话数据。
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(MossFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, line, fout) -> bool:
        """语言本身质量较高"""
        data = json.loads(line)
        text_list = []
        category = data["category"]
        conversation = data["conversation"]
        text_list.append("以下是一组任务类型为{}并且话题相关的对话：".format(category))
        for temp in conversation:
            q = self.zh_process(temp["human"])
            a = self.zh_process(temp["assistant"])
            text_list.append("问题：\n{}\n回答：{}".format(q, a))
        text = "\n".join(text_list)
        if not self.quality_assurance(text):
            return False
        d = {"text": text}
        fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return True
                    
class WikiCNFormatHandler(FormatHandler):
    """example data
    [
        {
            "completion": "昭通机场（ZPZT）是位于中国云南昭通的民用机场，始\",
            "source": "wikipedia.zh2307"
        },
        {
            "completion": "佐洛奇夫（Zolochiv），或按俄语译为佐洛乔夫（Золочев），是乌克兰西部利沃夫州佐洛奇夫区Золочівська міська громада内的城市，为该区及该市镇的行政中心。",
            "source": "wikipedia.zh2307"
        }
    ]
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(WikiCNFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def process_one_file(self, file_path):
        line_count = 0
        jump_count = 0
        with open(self.output_path, "a") as fout:
            with open(self.input_path + "/" + file_path, "r", encoding="utf-8") as fin:
                data = json.load(fin)
                # for item in tqdm(data):
                for item in data:
                    line_count += 1
                    if not self.quality_assurance(self.zh_process(item["completion"])):
                        jump_count += 1
                        continue
                    d = {"text": self.zh_process(item["completion"])}
                    fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return line_count, jump_count

class ZhihuFormatHandler(FormatHandler):
    """example data
    column: INSTRUCTION, RESPONSE, SOURCE, METADATA
    """
    def __init__(self, input_path, output_path, dataset_name):
        super(ZhihuFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def get_file_list(self) -> list:
        """或取输入路径下的全部文件，可以放一些文件名判断逻辑"""
        files = os.listdir(self.input_path)
        files = [i for i in files if ".parquet" in i]
        return files
    def process_one_file(self, file_path):
        line_count = 0
        jump_count = 0
        
        with open(self.output_path, "a") as fout:
            table = pq.read_table(self.input_path + "/" + file_path)  # encoding?
            for row in table.to_pydict():
                line_count += 1
                instruction = row["INSTRUCTION"]
                response = row["RESPONSE"]
                source = row["SOURCE"]
                metadata = row["METADATA"]
                text = self.zh_process("\n".join("指令：", instruction, "回答：", response))
                if not self.quality_assurance(text):
                    jump_count += 1
                    continue
                d = {"text": text}
                fout.write(json.dumps(d, ensure_ascii = False) + "\n")
        return line_count, jump_count

def test_run():
    """简单测试"""
    script_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_directory)
    output_path_root = script_directory + "/output"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)
    
    dataset_handler = {
        "baidu_baike": BaiduBaikeFormatHandler,
        "baidu_QA": BaiduQAFormatHandler,
        "BELLE": BELLEFormatHandler,
        "BELLE_conversations": BELLEConversationsFormatHandler,
        "firefly": FireflyFormatHandler,
        "moss": MossFormatHandler,
        "zhihu": ZhihuFormatHandler,
        "wiki_cn": WikiCNFormatHandler
    }
    dataset_list = list(dataset_handler.keys())

    for dataset_name, Handler in dataset_handler.items():
        input_path = dataset_name
        output_path = output_path_root + "/processed_{}.jsonl".format(dataset_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all()
    
def main_run():
    input_path_root = "/data/step0_raw"
    output_path_root = "/data/step1_unified_format"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    dataset_process_info = {
        "baidu_baike": (input_path_root + "/baidu_baike", BaiduBaikeFormatHandler),
        "baidu_QA": (input_path_root + "/baidu_wenda", BaiduQAFormatHandler),
        "BELLE_2M": (input_path_root + "/BELLE/2M", BELLEFormatHandler),
        "BELLE_3_5M": (input_path_root + "/BELLE/3_5M", BELLEConversationsFormatHandler),
        "firefly": (input_path_root + "/firefly/train_1M", FireflyFormatHandler),
        "moss": (input_path_root + "/moss", MossFormatHandler),
        "zhihu": (input_path_root + "/zhihu/data", ZhihuFormatHandler),
        "wiki_cn": (input_path_root + "/wiki_cn", WikiCNFormatHandler)
    }

    for dataset_name, info in dataset_process_info.items():
        input_path = info[0]
        Handler = info[1]
        output_path = output_path_root + "/processed_{}.jsonl".format(dataset_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all()


if __name__ == "__main__":
    test_mode = True
    if test_mode:
        test_run()
    else: 
        main_run()



    


    
