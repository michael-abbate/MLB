/*create table mlb_combined as*/
select a.ab_id, a.g_id,  a.batter_id, a.pitcher_id,  /* IDs */
g.home_team,
g.away_team,
a.inning,
a.p_score as pitchers_score, /* score for pitchers team*/
p.b_score as batters_score,
a.p_throws, /* handedness of pitcher*/
a.stand as "batters_stance", /* righty/lefty batters*/
case  
	when top = 'True' Then 'Away_Batter'
	when top = 'False' Then 'Home_Batter'
END as "Batter_HomeAway",
max(p.pitch_num) as "Final_Pitch_Count_Of_At_Bat",
p.pitch_type,
p.start_speed,
p.b_count as "balls",
p.s_count as "strikes",
p.outs,
p.on_1b,
p.on_2b,
p.on_3b,
g.weather,
a.event
from atbats a, games g, pitches p
where g.g_id = a.g_id
and a.ab_id = p.ab_id
group by a.ab_id;

