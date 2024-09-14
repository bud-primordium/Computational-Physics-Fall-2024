!找出二十四点的表达式
!四张纸牌的值从键盘输入
!!本程序采用的方法为枚举法，通过遍历所有的组合情况，输出满足条件的情况
!!0.1 先抽取四个数中的两个数进行运算，得到的结果加上剩余的一个数就剩三个数
!!0.2 再从这三个数中抽取两个进行运算，得到的结果加上剩余的一个数就剩两个数
!!0.3 对剩下的两个数进行运算，判断结果是否为24，若满足，则输出结果
program point_24
implicit none
    integer i, j, k, l, m, n, p, q, s, t !用于遍历所有可能的工作变量
    integer a, b, c, d !代表算式中的第一、二、三、四位数
    real x, y, z !存储第一次计算后剩余三个数的值
    character(len=2) aa, bb, cc, dd !代表算式中第一、二、三、四位数的字符串

    integer, dimension(4) :: num !存储最初输入的四个值
    integer, dimension(4) :: num_work !将num中的值存入其中，避免工作时抹去num中的数据
    integer, dimension(2) :: remains !存储先选取两个数后剩下的两个数
    
    real, dimension(3) :: three_num !存储第一次计算后剩余的三个数
    real, dimension(2) :: two_num   !存储第二次计算后剩余的两个数
    real, dimension(6) :: fir_cal !存储第一次计算后的结果
    real, dimension(6) :: sec_cal !存储第二次计算后的结果
    real, dimension(6) :: thi_cal !存储第三次计算后的结果

    character(len=20):: three_num_cha(3) !存储第一次计算后三个数对应的表达式
    character(len=20):: two_num_cha(2) !存储第二次计算后两个数对应的表达式
    character(len=20):: fir_cal_cha(6) !存储第一次计算后两个数对应的算式
    character(len=20):: sec_cal_cha(6) !存储第二次计算后两个数对应的算式
    character(len=20):: thi_cal_cha(6) !存储第三次计算后两个数对应的算式

    print*, "Please input four numbers(range 1~13):"
    read*, num(1), num(2), num(3), num(4)
!!1.最外两层循环（使用i,j作为参数）用于遍历“从四个数中选两个数”的所有6种可能，从四个数中抽取两个进行运算
    do i = 1, 3
        do j = i + 1, 4
            num_work = num !将原数组的值赋给工作数组，防止操作过程中原数组中的值被抹去
            a = num_work(i) !将这一次取出的两个数存储在a,b中
            b = num_work(j)
            num_work(i) = 0 !将已经取出的数字设为零，便于下方代码块中操作
            num_work(j) = 0
            !下面这一代码块可以取出原数组中除a，b之外的两个数
            l = 1
            do k = 1, 4
                if (num_work(k) /= 0) then
                    remains(l) = num_work(k)
                    l = l + 1
                end if
            end do

            c = remains(1)
            d = remains(2)
            !line32-line47的工作是将遍历取出的两个数存入a,b中，剩余两个数存入c,d中
            !下面将a,b,c,d转换为字符串
            write(aa, '(I2)') a
            write(bb, '(I2)') b 
            write(cc, '(I2)') c
            write(dd, '(I2)') d
            !进行第一次运算，得到的结果存入fir_cal中
            fir_cal(1) = a + b
            fir_cal(2) = a - b
            fir_cal(3) = b - a
            fir_cal(4) = a * b
            fir_cal(5) = (1.0 * a) / b
            fir_cal(6) = (1.0 * b) / a
            !将第一次运算对应的表达式存入fir_cal_cha中
            fir_cal_cha(1) = '(' // aa // '+' // bb // ')'
            fir_cal_cha(2) = '(' // aa // '-' // bb // ')'
            fir_cal_cha(3) = '(' // bb // '-' // aa // ')'
            fir_cal_cha(4) = '(' // aa // '*' // bb // ')'
            fir_cal_cha(5) = '(' // aa // '/' // bb // ')'
            fir_cal_cha(6) = '(' // bb // '/' // aa // ')'
!!2.下面这一层循环（使用m作为参数）用于挑出fir_cal中的元素，将它和c,d这三个数进行运算
            do m = 1, 6
                three_num(1) = fir_cal(m) !挑出fir_cal中的一个数，并与c,d组成一个新数组three_num
                three_num(2) = c
                three_num(3) = d

                three_num_cha(1) = fir_cal_cha(m) !将three_num中三个元素对应的表达式存储到three_num_cha中
                three_num_cha(2) = cc
                three_num_cha(3) = dd
!!3.下面的两层循环（使用p,q作为参数）用于遍历“从三个数中抽取两个”的所有可能，从三个数中抽取两个进行运算。运算结果加上剩余的一个数就剩两个数
                do p = 1, 2
                    do q = p + 1, 3
                        x = three_num(p) !将这一次取出的两个值存储在x,y中
                        y = three_num(q)
                        !进行第二次运算，得到的结果存入sec_cal中
                        sec_cal(1) = x + y
                        sec_cal(2) = x - y
                        sec_cal(3) = y - x
                        sec_cal(4) = x * y
                        sec_cal(5) = (1.0 * x) / y
                        sec_cal(6) = (1.0 * y) / x 
                        !将第二次运算对应的表达式存入sec_cal_cha中
                        sec_cal_cha(1) = '(' // trim(three_num_cha(p)) // '+' // trim(three_num_cha(q)) // ')'
                        sec_cal_cha(2) = '(' // trim(three_num_cha(p)) // '-' // trim(three_num_cha(q)) // ')'
                        sec_cal_cha(3) = '(' // trim(three_num_cha(q)) // '-' // trim(three_num_cha(p)) // ')'
                        sec_cal_cha(4) = '(' // trim(three_num_cha(p)) // '*' // trim(three_num_cha(q)) // ')'
                        sec_cal_cha(5) = '(' // trim(three_num_cha(p)) // '/' // trim(three_num_cha(q)) // ')'
                        sec_cal_cha(6) = '(' // trim(three_num_cha(q)) // '/' // trim(three_num_cha(p)) // ')'
!!4.下面的一层循环（使用t为参数）用于挑出sec_cal中的元素，将它和剩余的一个数进行计算
                        do t = 1, 6
                            two_num(1) = sec_cal(t) !挑出sec_cal中的一个数，并与剩下的一个数组成新数组two_num
                            two_num(2) = three_num(6 - q - p)

                            two_num_cha(1) = sec_cal_cha(t) !将two_num中三个元素对应的表达式存储到two_num_cha
                            two_num_cha(2) = three_num_cha(6-p-q)
                            !进行第三次运算，得到的结果存入thi_cal中
                            thi_cal(1) = two_num(1) + two_num(2)
                            thi_cal(2) = two_num(1) - two_num(2)
                            thi_cal(3) = two_num(2) - two_num(1)
                            thi_cal(4) = two_num(1) * two_num(2)
                            thi_cal(5) = two_num(1) / two_num(2)
                            thi_cal(6) = two_num(2) / two_num(1)
                            !将第三次运算对应的表达式存入thi_cal_cha中
                            thi_cal_cha(1) = "(" // trim(two_num_cha(1)) // "+" // trim(two_num_cha(2)) // ')'
                            thi_cal_cha(2) = '(' // trim(two_num_cha(1)) // '-' // trim(two_num_cha(2)) // ')'
                            thi_cal_cha(3) = '(' // trim(two_num_cha(2)) // '-' // trim(two_num_cha(1)) // ')'
                            thi_cal_cha(4) = '(' // trim(two_num_cha(1)) // '*' // trim(two_num_cha(2)) // ')'
                            thi_cal_cha(5) = '(' // trim(two_num_cha(1)) // '/' // trim(two_num_cha(2)) // ')'
                            thi_cal_cha(6) = '(' // trim(two_num_cha(2)) // '/' // trim(two_num_cha(1)) // ')'
!!5.下面一层循环（使用s为参数）挑出thi_cal中的元素，将三次运算之后的结果与24进行比较，如果等于24，则输出表达式
                            do s = 1, 6
                                if (abs(thi_cal(s) - 24) < 0.001) then !由于计算机采用浮点运算，当二者差值小于0.001时就认为二者相等
                                   print*, thi_cal_cha(s)
                                end if
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
end program point_24