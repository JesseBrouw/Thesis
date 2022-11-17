;Copyright 2018 The MITRE Corporation. All rights reserved. Approved for public release. Distribution unlimited 17-2122.
; For more information on CALDERA, the automated adversary emulation system, visit https://github.com/mitre/caldera or email attack@mitre.org
; This has 5 hosts, 1 user, 1 admin per host, 1 active account per host
(define (problem p5_hosts_trial_1)
(:domain caldera)
(:objects
    id_adomain - observeddomain
    id_ddomaincredential - observeddomaincredential
    id_cdomainuser - observeddomainuser
    id_bihost id_bbhost id_uhost id_nhost id_ghost - observedhost
    num__15 num__8 num__23 num__9 num__37 num__29 num__30 num__36 num__16 num__22 - num
    id_byshare id_bxshare id_cashare id_bwshare id_bzshare - observedshare
    id_btschtask id_bsschtask id_brschtask id_buschtask id_bvschtask - observedschtask
    id_bctimedelta id_otimedelta id_bjtimedelta id_vtimedelta id_htimedelta - observedtimedelta
    id_ccfile id_cffile id_cefile id_cdfile id_cbfile - observedfile
    str__k str__bh str__bq str__r str__t str__bf str__ba str__m str__alpha str__z str__bo str__bm str__l str__s str__y str__bg str__bn str__james str__e str__f str__b - string
    id_cgrat id_chrat id_cirat id_ckrat id_cjrat id_bprat - observedrat
)
(:init
    (knows id_bprat)
    (knows id_nhost)
    (knows_property id_bprat pexecutable)
    (knows_property id_bprat phost)
    (knows_property id_nhost pfqdn)
    (MEM_CACHED_DOMAIN_CREDS id_bbhost id_ddomaincredential)
    (MEM_CACHED_DOMAIN_CREDS id_bihost id_ddomaincredential)
    (MEM_CACHED_DOMAIN_CREDS id_ghost id_ddomaincredential)
    (MEM_CACHED_DOMAIN_CREDS id_nhost id_ddomaincredential)
    (MEM_CACHED_DOMAIN_CREDS id_uhost id_ddomaincredential)
    (MEM_DOMAIN_USER_ADMINS id_bbhost id_cdomainuser)
    (MEM_DOMAIN_USER_ADMINS id_bihost id_cdomainuser)
    (MEM_DOMAIN_USER_ADMINS id_ghost id_cdomainuser)
    (MEM_DOMAIN_USER_ADMINS id_nhost id_cdomainuser)
    (MEM_DOMAIN_USER_ADMINS id_uhost id_cdomainuser)
    (mem_hosts id_adomain id_bbhost)
    (mem_hosts id_adomain id_bihost)
    (mem_hosts id_adomain id_ghost)
    (mem_hosts id_adomain id_nhost)
    (mem_hosts id_adomain id_uhost)
    (prop_cred id_cdomainuser id_ddomaincredential)
    (PROP_DC id_bbhost no)
    (PROP_DC id_bihost no)
    (PROP_DC id_ghost no)
    (PROP_DC id_nhost no)
    (PROP_DC id_uhost no)
    (PROP_DNS_DOMAIN id_adomain str__b)
    (PROP_DNS_DOMAIN_NAME id_bbhost str__bf)
    (PROP_DNS_DOMAIN_NAME id_bihost str__bm)
    (PROP_DNS_DOMAIN_NAME id_ghost str__k)
    (PROP_DNS_DOMAIN_NAME id_nhost str__r)
    (PROP_DNS_DOMAIN_NAME id_uhost str__y)
    (PROP_DOMAIN id_bbhost id_adomain)
    (PROP_DOMAIN id_bihost id_adomain)
    (PROP_DOMAIN id_cdomainuser id_adomain)
    (PROP_DOMAIN id_ddomaincredential id_adomain)
    (PROP_DOMAIN id_ghost id_adomain)
    (PROP_DOMAIN id_nhost id_adomain)
    (PROP_DOMAIN id_uhost id_adomain)
    (prop_elevated id_bprat yes)
    (prop_executable id_bprat str__bq)
    (PROP_FQDN id_bbhost str__bg)
    (PROP_FQDN id_bihost str__bn)
    (PROP_FQDN id_ghost str__l)
    (PROP_FQDN id_nhost str__s)
    (PROP_FQDN id_uhost str__z)
    (prop_host id_bctimedelta id_bbhost)
    (prop_host id_bjtimedelta id_bihost)
    (prop_host id_bprat id_nhost)
    (prop_host id_htimedelta id_ghost)
    (prop_host id_otimedelta id_nhost)
    (prop_host id_vtimedelta id_uhost)
    (PROP_HOSTNAME id_bbhost str__bh)
    (PROP_HOSTNAME id_bihost str__bo)
    (PROP_HOSTNAME id_ghost str__m)
    (PROP_HOSTNAME id_nhost str__t)
    (PROP_HOSTNAME id_uhost str__ba)
    (PROP_IS_GROUP id_cdomainuser no)
    (PROP_MICROSECONDS id_bctimedelta num__30)
    (PROP_MICROSECONDS id_bjtimedelta num__37)
    (PROP_MICROSECONDS id_htimedelta num__9)
    (PROP_MICROSECONDS id_otimedelta num__16)
    (PROP_MICROSECONDS id_vtimedelta num__23)
    (PROP_PASSWORD id_ddomaincredential str__e)
    (PROP_SECONDS id_bctimedelta num__29)
    (PROP_SECONDS id_bjtimedelta num__36)
    (PROP_SECONDS id_htimedelta num__8)
    (PROP_SECONDS id_otimedelta num__15)
    (PROP_SECONDS id_vtimedelta num__22)
    (PROP_SID id_cdomainuser str__f)
    (PROP_TIMEDELTA id_bbhost id_bctimedelta)
    (PROP_TIMEDELTA id_bihost id_bjtimedelta)
    (PROP_TIMEDELTA id_ghost id_htimedelta)
    (PROP_TIMEDELTA id_nhost id_otimedelta)
    (PROP_TIMEDELTA id_uhost id_vtimedelta)
    (PROP_USER id_ddomaincredential id_cdomainuser)
    (PROP_USERNAME id_cdomainuser str__james)
    (PROP_WINDOWS_DOMAIN id_adomain str__alpha)
)
(:goal
(and
    (prop_host id_ckrat id_ghost)
    (prop_host id_cjrat id_bihost)
    (prop_host id_cgrat id_bbhost)
    (prop_host id_chrat id_uhost)
)
)
)