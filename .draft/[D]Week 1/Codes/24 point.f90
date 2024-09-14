program game24_pro
    implicit none

    ! 声明变量
    integer :: maxn
    integer, parameter :: max_limit = 8
    real, allocatable :: numbers(:)
    character(len=200), allocatable :: expressions(:)
    integer :: i
    logical :: found_solution

    ! 提示用户输入数字的个数
    print *, '请输入数字的个数（1到8）：'
    read *, maxn

    ! 验证输入的数字个数是否在有效范围内
    if (maxn < 1 .or. maxn > max_limit) then
        print *, '错误：数字个数必须在1到8之间。程序终止。'
        stop
    end if

    ! 分配数组大小
    allocate(numbers(maxn))
    allocate(expressions(maxn))

    ! 提示用户输入具体的数字
    print *, '请输入', maxn, '个1到13之间的数字（用回车分隔，超范围不保证精度）：'
    do i = 1, maxn
        read *, numbers(i)
        ! 验证每个数字是否在1到48之间
        if (numbers(i) < -5 .or. numbers(i) > 24) then
            print *, '错误：数字必须在-5到24之间。程序终止。'
            deallocate(numbers)
            deallocate(expressions)
            stop
        end if
        ! 将数字转换为字符串表达式
        write(expressions(i), '(F0.2)') numbers(i)
    end do

    ! 初始化解决方案标志
    found_solution = .false.

    ! 调用解决24点的子程序
    call solve_24(numbers, expressions, found_solution)

    ! 根据是否找到解决方案进行相应输出
    if (.not. found_solution) then
        print *, '没有找到满足条件的解。'
    end if

    ! 释放分配的内存
    deallocate(numbers)
    deallocate(expressions)

  contains
  
    recursive subroutine solve_24(nums, exprs, found)
      implicit none
      real, intent(in) :: nums(:)
      character(len=200), intent(in) :: exprs(:)
      logical, intent(inout) :: found
      integer :: n
      integer :: i, j, k
      real :: a, b, result
      real, allocatable :: new_nums(:)
      character(len=200), allocatable :: new_exprs(:)
      character(len=200) :: expr_a, expr_b, new_expr
      character(len=1), parameter :: ops(4) = (/ '+', '-', '*', '/' /)
  
      n = size(nums)
  
      if (found) return
  
      if (n == 1) then
         if (abs(nums(1) - 24.0) < 1e-4) then

            print *, '找到一个解：', trim(exprs(1)), nums(1)
            found = .true.
         end if
         return
      end if
  
      do i = 1, n - 1
         do j = i + 1, n
  
            a = nums(i)
            b = nums(j)
            expr_a = exprs(i)
            expr_b = exprs(j)
  
            do k = 1, 4  ! 遍历四种运算符
  
               ! 处理 a op b 的情况
               select case(k)
               case(1)
                  result = a + b
                  new_expr = '(' // trim(expr_a) // '+' // trim(expr_b) // ')'
               case(2)
                  result = a - b
                  new_expr = '(' // trim(expr_a) // '-' // trim(expr_b) // ')'
               case(3)
                  result = a * b
                  new_expr = '(' // trim(expr_a) // '*' // trim(expr_b) // ')'
               case(4)
                  if (abs(b) < 1e-6) then
                     cycle  ! 避免除以零
                  end if
                  result = a / b
                  new_expr = '(' // trim(expr_a) // '/' // trim(expr_b) // ')'
               end select
  
               call remove_two(nums, exprs, i, j, new_nums, new_exprs)
               new_nums(size(new_nums)) = result
               new_exprs(size(new_exprs)) = new_expr
               call solve_24(new_nums, new_exprs, found)
               if (found) then
                  deallocate(new_nums)
                  deallocate(new_exprs)
                  return
               end if
  
               ! 处理 b op a 的情况（对于减法和除法）
               if (k == 2 .or. k == 4) then
                  select case(k)
                  case(2)
                     result = b - a
                     new_expr = '(' // trim(expr_b) // '-' // trim(expr_a) // ')'
                  case(4)
                     if (abs(a) < 1e-6) then
                        cycle
                     end if
                     result = b / a
                     new_expr = '(' // trim(expr_b) // '/' // trim(expr_a) // ')'
                  end select
  
                  call remove_two(nums, exprs, i, j, new_nums, new_exprs)
                  new_nums(size(new_nums)) = result
                  new_exprs(size(new_exprs)) = new_expr
                  call solve_24(new_nums, new_exprs, found)
                  if (found) then
                     deallocate(new_nums)
                     deallocate(new_exprs)
                     return
                  end if
               end if
  
            end do  ! 运算符循环结束
  
            deallocate(new_nums)
            deallocate(new_exprs)
  
         end do  ! j 循环结束
      end do  ! i 循环结束
  
    end subroutine solve_24
  
    subroutine remove_two(nums, exprs, idx1, idx2, new_nums, new_exprs)
      implicit none
      real, intent(in) :: nums(:)
      character(len=200), intent(in) :: exprs(:)
      integer, intent(in) :: idx1, idx2
      real, allocatable, intent(out) :: new_nums(:)
      character(len=200), allocatable, intent(out) :: new_exprs(:)
      integer :: i, j, n
  
      n = size(nums)
      allocate(new_nums(n - 1))
      allocate(new_exprs(n - 1))
  
      j = 0
      do i = 1, n
         if (i /= idx1 .and. i /= idx2) then
            j = j + 1
            new_nums(j) = nums(i)
            new_exprs(j) = exprs(i)
         end if
      end do
  
    end subroutine remove_two
  
  end program game24_pro