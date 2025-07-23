import sys
sys.path.append("")

import edge_tts

async def generate_audio(text,outputFilename):
    communicate = edge_tts.Communicate(text,"vi-VN-NamMinhNeural") #vi-VN-HoaiMyNeural #vi-VN-NamMinhNeural
    await communicate.save(outputFilename)


if __name__ == "__main__":
    import asyncio
    text = """Bạn có biết, cuốn sách "Đắc Nhân Tâm" đã giúp triệu triệu người thay đổi cuộc đời? Nhưng cuộc hành trình chinh phục lòng người không hề dễ dàng…  
    Chúng ta cùng theo dõi câu chuyện của một người làm nghề bán hàng, Minh, luôn tràn đầy hoài bão. Anh thấy rằng, trong thế giới cạnh tranh, việc kết nối với khách hàng là rất quan trọng. Anh quyết định đọc "Đắc Nhân Tâm" để hiểu rõ hơn về tâm lý con người.  
    Mục tiêu của Minh rất rõ ràng: muốn trở thành nhân viên xuất sắc, tăng doanh số và ghi dấu ấn trong lòng mỗi khách hàng. Nhưng thực tế thì không như mơ... Anh đối mặt với sự lạnh lùng của những khách hàng khó tính. Không khí im lặng, ánh mắt xa lạ khiến Minh bắt đầu chùn bước.  
    Giữa lúc tuyệt vọng, Minh nhớ lại một trong những bài học quý giá: “Hãy quan tâm đến người khác như bạn muốn được quan tâm.” Anh quyết định áp dụng điều đó.  
    Ngày hôm sau, Minh cố gắng lắng nghe và thấu hiểu tâm sự của từng khách hàng. Thế rồi, bất ngờ xảy ra! Khách hàng bắt đầu cười, chia sẻ và không chỉ mua sản phẩm, họ còn muốn nghe thêm những câu chuyện của Minh.  
    Chỉ sau một thời gian ngắn, doanh số của Minh đã tăng vọt!  
    Bài học rút ra từ câu chuyện này thật đơn giản, nhưng sâu sắc: Kết nối với con người là chìa khóa mở mọi cánh cửa thành công.  
    Vậy bạn có bao giờ thử thấu hiểu tâm tư của người khác chưa"""
    outputFilename = "output/audio_tts_test.wav"

    asyncio.run(generate_audio(text, outputFilename))
    print(f"Audio saved to {outputFilename}")




