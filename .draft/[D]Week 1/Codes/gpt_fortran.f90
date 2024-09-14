program scheme1_multiple_solutions
    implicit none
    integer, parameter :: N = 150
    integer(8) :: a, b, c, d, e, i
    integer(8) :: sum_abc, e5, d5, difference
    type :: triple
      integer(8) :: a, b, c
    end type triple
    type(triple), allocatable :: triples(:)
    integer(8), allocatable :: keys(:)
    integer(8) :: key_count
    integer(8) :: num_solutions
    type(triple), allocatable :: solution_triples(:)
    integer(8), allocatable :: solution_de(:,:)
  
    ! Step 1: 构建三次幂和表 (a^5 + b^5 + c^5)
    integer(8) :: num_entries
    integer(8), dimension(:), allocatable :: sums
    type(triple), dimension(:), allocatable :: entries
  
    ! 用哈希表存储三次幂和
    integer(8) :: max_sum
    integer(8), dimension(:), allocatable :: hash_table
    integer(8), dimension(:), allocatable :: hash_count
  
    ! 初始化哈希表大小 (调整类型为 INTEGER(8))
    max_sum = int(N, 8)**5 * 3
    allocate(hash_table(0:max_sum), hash_count(0:max_sum))
    hash_table = -1
    hash_count = 0
  
    num_entries = 0
    do a = 1, N-1
      do b = a, N-1
        do c = b, N-1
          sum_abc = a**5 + b**5 + c**5
          if (hash_table(sum_abc) == -1) then
            ! 新键，分配并增加条目数
            hash_table(sum_abc) = hash_count + 1
            hash_count = hash_count + 1
            num_entries = num_entries + 1
            allocate(entries(num_entries))
            entries(num_entries)%a = a
            entries(num_entries)%b = b
            entries(num_entries)%c = c
          else
            ! 键已经存在，增加组合
            num_entries = num_entries + 1
            allocate(entries(num_entries))
            entries(num_entries)%a = a
            entries(num_entries)%b = b
            entries(num_entries)%c = c
          end if
        end do
      end do
    end do
  
    ! Step 2: 使用两个循环进行查找 (e 和 d)
    num_solutions = 0
    do e = 1, N-1
      e5 = e**5
      do d = 1, e
        d5 = d**5
        difference = e5 - d5
        if (difference >= 0 .and. hash_table(difference) /= -1) then
          ! 在哈希表中找到差值，存储解决方案
          num_solutions = num_solutions + 1
          allocate(solution_triples(num_solutions))
          allocate(solution_de(2, num_solutions))
          solution_triples(num_solutions) = entries(hash_table(difference))
          solution_de(1, num_solutions) = d
          solution_de(2, num_solutions) = e
        end if
      end do
    end do
  
    ! 输出结果
    do i = 1, num_solutions
      print *, "Solution: ", solution_triples(i)%a, solution_triples(i)%b, solution_triples(i)%c, " | d: ", solution_de(1, i), " e: ", solution_de(2, i)
    end do
  
    ! 释放分配的内存
    deallocate(entries)
    deallocate(solution_triples)
    deallocate(solution_de)
    deallocate(hash_table)
  
  end program scheme1_multiple_solutions
  