program power_sum_schemes
    implicit none
    integer, parameter :: N = 250
    integer, dimension(:,:), allocatable :: triple_power_sum, double_power_sum
    integer, dimension(:,:), allocatable :: triple_indices, double_indices
    integer :: n_triples, n_doubles
    integer :: i, j, k, a, b, c, d, e
    real :: start_time, end_time
    integer :: n_solutions1, n_solutions2
  
    ! Scheme 1
    call cpu_time(start_time)
    call scheme1_multiple_solutions(N, n_solutions1)
    call cpu_time(end_time)
    print *, 'Scheme 1 found ', n_solutions1, ' solutions in ', end_time - start_time, ' seconds.'
  
    ! Scheme 2
    call cpu_time(start_time)
    call scheme2_multiple_solutions(N, n_solutions2)
    call cpu_time(end_time)
    print *, 'Scheme 2 found ', n_solutions2, ' solutions in ', end_time - start_time, ' seconds.'
  
  contains
  
    ! Scheme 1: Triple power sum search
    subroutine scheme1_multiple_solutions(N, n_solutions)
      integer, intent(in) :: N
      integer, intent(out) :: n_solutions
      integer, dimension(N**3) :: triple_sums
      integer :: a, b, c, d, e, i, sum_abc, e5, d5, difference
      n_solutions = 0
  
      ! Initialize triple power sum array
      do a = 1, N - 1
        do b = a, N - 1
          do c = b, N - 1
            sum_abc = a**5 + b**5 + c**5
            triple_sums(sum_abc) = triple_sums(sum_abc) + 1
          end do
        end do
      end do
  
      ! Search for solutions using two loops (e, d)
      do e = 1, N - 1
        e5 = e**5
        do d = 1, e - 1
          d5 = d**5
          difference = e5 - d5
          if (triple_sums(difference) > 0) then
            n_solutions = n_solutions + triple_sums(difference)
          end if
        end do
      end do
    end subroutine scheme1_multiple_solutions
  
    ! Scheme 2: Double power sum search
    subroutine scheme2_multiple_solutions(N, n_solutions)
      integer, intent(in) :: N
      integer, intent(out) :: n_solutions
      integer, dimension(N**2) :: double_sums
      integer :: a, b, c, d, e, i, sum_ab, e5, d5, difference
      n_solutions = 0
  
      ! Initialize double power sum array
      do a = 1, N - 1
        do b = a, N - 1
          sum_ab = a**5 + b**5
          double_sums(sum_ab) = double_sums(sum_ab) + 1
        end do
      end do
  
      ! Search for solutions using three loops (e, d, c)
      do e = 1, N - 1
        e5 = e**5
        do d = 1, e - 1
          d5 = d**5
          do c = 1, d - 1
            difference = e5 - d5 - c**5
            if (double_sums(difference) > 0) then
              n_solutions = n_solutions + double_sums(difference)
            end if
          end do
        end do
      end do
    end subroutine scheme2_multiple_solutions
  
  end program power_sum_schemes