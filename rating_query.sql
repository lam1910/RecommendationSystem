/* sql query to select all record on each user group with more than 60% of user rate the item >= 4 and put it into table 
with name followed group_group-name_good_rating*/

-- group 1
create table groupdata.group_one_good_rating as(
	with tmp1 as (select * from groupdata.group_one where rating >= 4)
	, tmp2 as (select movieid, count(userid) as no_of_rating from groupdata.group_one group by (movieid)) 
	,tmp3 as (select movieid, count(userid) as no_of_good_rating from tmp1 group by (movieid))
	select tmp1.* from tmp1, tmp2, tmp3
	where tmp1.movieid = tmp2.movieid and tmp1.movieid = tmp3.movieid and tmp3.no_of_good_rating >= 0.6 * tmp2.no_of_rating
);

-- group 2
create table groupdata.group_two_good_rating as(
	with tmp1 as (select * from groupdata.group_two where rating >= 4)
	, tmp2 as (select movieid, count(userid) as no_of_rating from groupdata.group_two group by (movieid)) 
	,tmp3 as (select movieid, count(userid) as no_of_good_rating from tmp1 group by (movieid))
	select tmp1.* from tmp1, tmp2, tmp3
	where tmp1.movieid = tmp2.movieid and tmp1.movieid = tmp3.movieid and tmp3.no_of_good_rating >= 0.6 * tmp2.no_of_rating
);

--group 3
create table groupdata.group_three_good_rating as(
	with tmp1 as (select * from groupdata.group_three where rating >= 4)
	, tmp2 as (select movieid, count(userid) as no_of_rating from groupdata.group_three group by (movieid)) 
	,tmp3 as (select movieid, count(userid) as no_of_good_rating from tmp1 group by (movieid))
	select tmp1.* from tmp1, tmp2, tmp3
	where tmp1.movieid = tmp2.movieid and tmp1.movieid = tmp3.movieid and tmp3.no_of_good_rating >= 0.6 * tmp2.no_of_rating
);
